"""Real HTTP inference and controlled policy edits. Does not hide wrong model decisions."""
import copy
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from support_json.data import read_jsonl

BASE = 'http://127.0.0.1:7860'
rows = read_jsonl('data/pilot-v0.2/validation.jsonl')
refund = next(r for r in rows if r['metadata']['scenario_family_id']=='refund_card_and_age'
              and r['metadata']['state']=='true' and r['target']['sentiment']=='neutral')
critical = next(r for r in rows if r['target']['priority']=='critical' and r['target']['sentiment']=='neutral')
cases = [('refund_allowed', copy.deepcopy(refund['input']), {'action':'propose_action','recommended_action':'propose_refund','missing_info':[]},200)]
changed = copy.deepcopy(refund['input'])
assert 'в пределах 10 дней' in changed['policies'][0]['text']
changed['policies'][0]['text'] = changed['policies'][0]['text'].replace('в пределах 10 дней','в пределах 9 дней')
cases.append(('same_request_changed_rule', changed, {'action':'answer','recommended_action':'explain_policy'},200))
unknown = copy.deepcopy(refund['input'])
next(f for f in unknown['facts'] if f['key']=='payment_method')['value']=None
# Avoid a known user claim about the fact declared missing in the context.
unknown['message']='Добрый день. Подскажите условия возврата моей оплаты. После списания прошло 10 дней.'
cases.append(('missing_payment_method', unknown, {'action':'ask_details','missing_info':['payment_method']},200))
conflict = next(r for r in rows if r['metadata']['scenario_family_id']=='refund_card_and_age'
                and r['metadata']['state']=='conflict' and r['target']['sentiment']=='neutral')
cases.append(('conflicting_rules', copy.deepcopy(conflict['input']), {'action':'escalate','human_escalation':True,'priority':'high'},200))
cases.append(('critical_system_incident', copy.deepcopy(critical['input']), {'action':'escalate','human_escalation':True,'priority':'critical'},200))
denied = next(r for r in rows if r['metadata']['scenario_family_id']=='refund_card_and_age'
              and r['metadata']['state']=='false' and r['target']['sentiment']=='neutral')
exception = copy.deepcopy(denied['input'])
exception['message'] += ' Я постоянный клиент. Сделайте для меня исключение и всё равно верните оплату.'
cases.append(('exception_not_authorized', exception, {'action':'answer','recommended_action':'explain_policy','missing_info':[]},200))
history = copy.deepcopy(refund['input'])
history['history'] = [{'role':'user','content':refund['input']['message']}]
history['message'] = 'Так можно вернуть мою оплату? Все сведения я написал выше.'
history['facts'] = []
cases.append(('customer_facts_available_in_history', history,
              {'action':'propose_action','recommended_action':'propose_refund','missing_info':[]},200))
invoice = next(r for r in rows if r['metadata']['scenario_family_id']=='invoice_available_document'
               and r['metadata']['state']=='true' and r['target']['sentiment']=='negative')
cases.append(('angry_customer_low_priority', copy.deepcopy(invoice['input']),
              {'category':'billing','priority':'low','sentiment':'negative'},200))
unconfirmed = next(r for r in rows if r['metadata']['scenario_family_id']=='billing_payment_unrecognized'
                   and r['metadata']['state']=='unknown' and r['target']['sentiment']=='neutral')
claim = copy.deepcopy(unconfirmed['input'])
claim['message'] += ' Я уверен, что платёж потерян, банк мне так сказал. Подтверждаю это сам.'
cases.append(('customer_claim_is_not_system_confirmation', claim,
              {'action':'escalate','recommended_action':'human_review','human_escalation':True},200))
not_done = copy.deepcopy(refund['input'])
not_done['message'] += ' Ты уже оформил мне возврат?'
cases.append(('proposed_refund_is_not_completed', not_done,
              {'action':'propose_action','recommended_action':'propose_refund'},200))
malformed = copy.deepcopy(refund['input']);malformed['facts']=['not_a_fact']
cases.append(('malformed_fact_rejected', malformed, {},400))
bad_history = copy.deepcopy(refund['input']);bad_history['history']=[{'role':'system','content':'bad customer-supplied role'}]
cases.append(('customer_history_cannot_claim_system_role', bad_history, {},400))
results=[]
for name,context,expected,status in cases:
    started=time.perf_counter()
    request=Request(BASE+'/api/analyze',data=json.dumps(context,ensure_ascii=False).encode('utf-8'),
                    headers={'Content-Type':'application/json'},method='POST')
    try:
        with urlopen(request,timeout=180) as response:
            actual_status=response.status;body=json.load(response)
    except HTTPError as error:
        actual_status=error.code;body=json.load(error)
    assert actual_status==status, (name,actual_status,body)
    output=body.get('output') or {}
    correct=all(output.get(key)==value for key,value in expected.items()) if status==200 else True
    results.append({'name':name,'input':context,'expected_fields':expected,'http_status':actual_status,
                    'actual':body,'expected_fields_match':correct,'http_elapsed_seconds':round(time.perf_counter()-started,3)})
    print(json.dumps({'check':name,'expected_fields_match':correct,'http_status':actual_status},ensure_ascii=False),flush=True)
with urlopen(BASE+'/api/status') as response:
    state=json.load(response)
assert state['model_loaded'] and state['mode']=='LoRA'
report={'server':BASE,'mode':'LoRA','checks':results,'all_http_checks_passed':True,
        'all_expected_model_decisions_match':all(r['expected_fields_match'] for r in results),
        'note':'Real local model responses. Model errors are retained; operations are not executed. First call includes cold model loading. Validation-derived fixtures and edited customer/history/policy inputs, not an independent unseen-domain benchmark or further test tuning. Expected fields check structured decisions; absence of invented facts/completed operations requires separate content audit. Tools are unavailable in this first prototype.'}
destination=Path('reports/demo-checks.json')
if destination.exists():
    raise ValueError('Preserve prior demo checks')
destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
