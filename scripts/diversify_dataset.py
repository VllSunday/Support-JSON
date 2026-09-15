"""Repair the uniform-condition bias using original domain rules and distinct fact keys."""
import copy
import json
import random
from collections import defaultdict
from pathlib import Path
import support_json.data as data
from support_json.catalog import scenarios as authored, leaf, REVIEW, DIAGNOSTICS

destination=Path('data/diverse-v0.5')
if destination.exists():
    raise ValueError('Dataset version already exists')
profiles=[{'company_id':f'diverse_cloud_{i:02d}','refund_days':i+2,'cancel_cutoff':i%12+1,
           'seats':i*3+2,'retention':i*5+10,'variant':i} for i in range(32)]
original_profiles=data.PROFILES
data.PROFILES={'train':profiles}

PHRASES={
    'refund':['Хочу вернуть деньги за сервис.','Можно вернуть оплату за подписку?'],
    'cancellation':['Отключите продление подписки, пожалуйста.','Хочу остановить подписку до следующего платежа.'],
    'access':['Не могу попасть в личный кабинет по паролю.','Пароль не подходит, помогите восстановить вход.'],
    'technical':['Сервис перестал открываться, помогите.','Приложение не отвечает, как разобраться?'],
    'billing':['Помогите разобраться с оплатой.','Нужно проверить информацию по платежу.'],
    'product':['Хочу поменять тариф для нашей команды.','Нужно изменить тариф под команду.'],
    'integration':['Перестала работать синхронизация интеграции.','Данные из подключённого сервиса не обновляются.']}

def varied(split,profile):
    items=authored(split,profile)
    for item in items:
        i=profile['variant']
        item['family']='diverse_'+item['family']
        # Billing has two distinct intents: preserve invoice versus duplicate wording.
        if item['category']!='billing':
            item['message']=PHRASES[item['category']][i%2]
        if item['category']=='technical':
            threshold=i+2
            item.update(condition=leaf('affected_user_count','ge',threshold,'tool_observation'),
                        true_facts={'affected_user_count':threshold},false_facts={'affected_user_count':threshold-1},
                        unknown_key='affected_user_count',fact_labels={'affected_user_count':'число затронутых пользователей'},
                        text=f'Массовый инцидент подтверждён, если система наблюдает affected_user_count не меньше {threshold}. '
                             'Для такого инцидента нужен человек; при меньшем числе затронутых пользователей предложить диагностику. '
                             'Если система не сообщила число, нужен человек. Срок исправления не обещать.',
                        true=REVIEW,false=DIAGNOSTICS,priority_true='critical',priority_false='high')
        else:
            # SLA belongs to the company, not to the emotional prefix/category alone.
            item['priority_true']=['low','medium','high'][i%3]
            item['priority_false']=['low','medium','high'][(i//3)%3]
    return items

data.scenarios=varied
candidates=data.build_rows()
data.PROFILES=original_profiles
original=data.read_jsonl('data/pilot-v0.2/train.jsonl')
seen={data.digest(data.semantic_input(r['input'])) for r in original}
unique=[]
for row in candidates:
    fp=data.digest(data.semantic_input(row['input']))
    if fp in seen:
        continue
    seen.add(fp)
    row['metadata'].update(annotation_version='0.5',source='authored_synthetic_domain_diversification')
    unique.append(row)
rng=random.Random(42)
rng.shuffle(unique)
strata=defaultdict(list)
for row in unique:
    strata[(row['metadata']['scenario_family_id'],row['metadata']['state'],row['target']['sentiment'])].append(row)
selected=list(original)
while len(selected)<512:
    advanced=False
    for key in sorted(strata):
        if strata[key] and len(selected)<512:
            selected.append(strata[key].pop())
            advanced=True
    if not advanced:
        raise ValueError('Insufficient independent candidates')
held=data.read_jsonl('data/pilot-v0.2/validation.jsonl')+data.read_jsonl('data/pilot-v0.2/test.jsonl')
audit=data.audit_rows(selected+held)
destination.mkdir()
data.write_jsonl(destination/'train.jsonl',selected)
data.write_jsonl(destination/'train.chatml.jsonl',[{'messages':data.messages(row['input'])+
    [{'role':'assistant','content':json.dumps(row['target'],ensure_ascii=False)}]} for row in selected])
audit.update(training_rows=512,purpose='validation_error_driven_domain_diversification',
    hypothesis='Uniform verified-age condition in v0.4 encourages unnecessary escalation; restore different domain facts and source requirements.',
    test_used_for_training_design=False,
    files_sha256={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.jsonl')})
(destination/'manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'n':512,'cross_split_overlap':audit['cross_split_overlap'],'priority_counts':audit['split_distributions']['train']['priority']},ensure_ascii=False))
