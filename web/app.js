'use strict';

const $ = id => document.getElementById(id);
let examples = [], revision = 0, busy = false, currentOutput = null;
const labels = {
  billing:'Оплата', access:'Доступ', technical:'Техническая проблема', refund:'Возврат',
  cancellation:'Отмена', integration:'Интеграция', product:'Продукт', other:'Другое',
  low:'Низкий', medium:'Средний', high:'Высокий', critical:'Критический',
  positive:'Положительное', neutral:'Нейтральное', negative:'Отрицательное',
  answer:'Ответить', ask_details:'Уточнить сведения', escalate:'Передать человеку',
  propose_action:'Предложить действие', explain_policy:'Объяснить правило',
  request_details:'Уточнить сведения', human_review:'Проверка человеком',
  propose_refund:'Предложить возврат', propose_cancellation:'Предложить отмену',
  propose_password_reset:'Предложить сброс пароля', propose_diagnostics:'Предложить диагностику',
  propose_invoice_download:'Предложить получение счёта', propose_plan_change:'Предложить смену тарифа',
  propose_integration_check:'Предложить проверку интеграции'
};
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text = (id, value) => { $(id).textContent = value; };
const translated = value => labels[value] || value;

function resetResult(stale = false) {
  revision++;
  const hadResult = currentOutput !== null || !$('errorState').hidden || !$('staleBanner').hidden;
  currentOutput = null;
  $('resultBody').hidden = true;
  $('errorState').hidden = true;
  $('timing').hidden = true;
  $('rawOut').hidden = true;
  $('rawToggle').hidden = true;
  $('rawToggle').setAttribute('aria-expanded', 'false');
  text('rawToggle', 'Показать сырой ответ');
  $('emptyState').hidden = busy;
  $('staleBanner').hidden = !(stale && (hadResult || busy));
  $('copyReply').disabled = $('copyJson').disabled = true;
}

function contextPreview() {
  for (const [field, listId, countId] of [['factsJson','factsList','factsCount'], ['historyJson','historyList','historyCount']]) {
    const list = $(listId);
    list.replaceChildren();
    try {
      const values = JSON.parse($(field).value || '[]');
      if (!Array.isArray(values)) throw Error();
      text(countId, values.length);
      for (const value of values) {
        const li = document.createElement('li');
        if (field === 'factsJson') {
          li.className = 'fact od-row-top';
          const unknown = value.value == null;
          const display = unknown ? 'значение неизвестно' : typeof value.value === 'object' ? JSON.stringify(value.value) : String(value.value);
          li.innerHTML = `<div class="fact__body od-field od-fill"><span class="fact__key">${esc(value.key || 'Без ключа')}</span><span class="fact__value${unknown?' fact__value--unknown':''}">${esc(display)}</span></div><span class="badge od-fixed ${value.origin==='tool_observation'?'badge--tool':'badge--user'}">${value.origin==='tool_observation'?'система':'слова клиента'}</span>`;
        } else {
          li.className = 'msg' + (value.role === 'assistant' ? ' msg--assistant' : '');
          li.innerHTML = `<span class="msg__role">${value.role==='assistant'?'Поддержка':'Клиент'}</span><p class="msg__text">${esc(value.content || '')}</p>`;
        }
        list.append(li);
      }
    } catch { text(countId, 'Ошибка JSON'); }
  }
  try {
    const rules = JSON.parse($('rules').value);
    const endings = {one:'правило',few:'правила',many:'правил',other:'правила'};
    text('rulesCount', Array.isArray(rules) ? `${rules.length} ${endings[new Intl.PluralRules('ru-RU').select(rules.length)]}` : 'Нужен список');
  } catch { text('rulesCount', 'Ошибка JSON'); }
}

function loadExample(index) {
  const example = examples[index];
  if (!example) return;
  $('example').value = index;
  $('appeal').value = example.input.message;
  $('rules').value = JSON.stringify(example.input.policies, null, 2);
  $('factsJson').value = JSON.stringify(example.input.facts, null, 2);
  $('historyJson').value = JSON.stringify(example.input.history || [], null, 2);
  for (const chip of $('exampleChips').querySelectorAll('input')) chip.checked = Number(chip.value) === index;
  text('exampleHint', 'Синтетический пример из validation. Факты и история относятся к этому обращению.');
  $('rulesError').hidden = true;
  $('rules').removeAttribute('aria-invalid');
  resetResult();
  contextPreview();
}

function showError(message, raw = null) {
  $('resultBody').hidden = $('emptyState').hidden = true;
  $('errorState').hidden = false;
  text('errorTitle', raw === null ? 'Разбор не выполнен' : 'Ответ не прошёл проверку');
  text('errorReason', message);
  text('rawOut', raw || '');
  $('rawToggle').hidden = raw === null;
}

function highlightJSON(value) {
  return esc(value).replace(/(&quot;(?:[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?)/g,
    (match, string, colon, literal, number) => {
      if (string) return `<span class="${colon?'json-key':'json-string'}">${string}</span>${colon || ''}`;
      return `<span class="${literal==='null'?'json-null':literal?'json-bool':'json-num'}">${match}</span>`;
    });
}

function showResult(output, policies, elapsed) {
  currentOutput = output;
  for (const [id, key] of [['valCategory','category'],['valSentiment','sentiment'],['valStrategy','action'],['valAction','recommended_action'],['valPriority','priority']]) text(id, translated(output[key]));
  text('valHandoff', output.human_escalation ? 'Нужна' : 'Не нужна');
  $('statPriority').dataset.level = output.priority;
  $('statHandoff').dataset.escalate = String(output.human_escalation);
  text('replyText', output.reply);
  text('actionNote', 'Это рекомендация. Приложение не выполняет операций и не отправляет ответ клиенту.');
  $('missingBlock').hidden = output.missing_info.length === 0;
  $('missingList').replaceChildren(...output.missing_info.map(value => {
    const li = document.createElement('li'); li.textContent = value; return li;
  }));
  $('rulesUsed').replaceChildren(...output.supported_rule_ids.map(id => {
    const li = document.createElement('li');
    li.className = 'od-row-top';
    li.innerHTML = `<code class="rule-id od-fixed">${esc(id)}</code><span class="rule-text od-fill">${esc(policies.find(p=>p.id===id)?.text || '')}</span>`;
    return li;
  }));
  if (!output.supported_rule_ids.length) {
    const li = document.createElement('li'); li.textContent = 'Модель не указала правила.'; $('rulesUsed').append(li);
  }
  $('jsonOut').innerHTML = highlightJSON(JSON.stringify(output, null, 2));
  text('timing', `${elapsed.toLocaleString('ru-RU')} с`);
  $('timing').hidden = false;
  $('resultBody').hidden = false;
  $('emptyState').hidden = $('errorState').hidden = $('staleBanner').hidden = true;
  $('copyReply').disabled = $('copyJson').disabled = false;
}

async function analyze() {
  if (busy) return;
  resetResult();
  $('rulesError').hidden = true;
  $('rules').removeAttribute('aria-invalid');
  let input;
  try {
    input = {message:$('appeal').value};
    for (const [id, key, name] of [['rules','policies','Правила'],['factsJson','facts','Факты'],['historyJson','history','История']]) {
      try { input[key] = JSON.parse($(id).value || (id==='rules'?'':'[]')); }
      catch {
        if (id === 'rules') {
          $('rulesError').hidden = false;
          text('rulesErrorText','Ошибка JSON: проверьте кавычки и запятые.');
          $('rules').setAttribute('aria-invalid','true');
        }
        throw Error(`${name}: ошибка JSON. Проверьте кавычки и запятые.`);
      }
    }
    if (!input.message.trim()) throw Error('Введите текст обращения.');
    if (!Array.isArray(input.policies) || !input.policies.length) throw Error('Добавьте правила списком JSON.');
    if (!Array.isArray(input.facts) || !Array.isArray(input.history)) throw Error('Факты и история должны быть списками JSON.');
  } catch (error) {
    showError(error instanceof SyntaxError ? 'В правилах, фактах или истории ошибка JSON. Проверьте кавычки и запятые.' : error.message);
    text('runStatus', 'Проверьте входные данные. Запрос к модели не отправлен.');
    return;
  }
  const requestRevision = revision;
  busy = true;
  $('runBtn').disabled = $('retryBtn').disabled = true;
  $('loadingState').hidden = false;
  $('emptyState').hidden = true;
  $('resultPane').setAttribute('aria-busy','true');
  text('runStatus', 'Модель разбирает обращение… Первый запрос включает загрузку весов.');
  try {
    const response = await fetch('/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(input)});
    const data = await response.json();
    if (revision !== requestRevision) {
      text('runStatus', 'Данные изменились. Ответ предыдущего запроса скрыт; выполните новый разбор.');
      return;
    }
    if (!response.ok) throw Error(data.error || 'Сервер не выполнил запрос.');
    if (data.errors.length || data.output === null) {
      showError('Формат или согласованность полей нарушены. Черновик скрыт: передайте обращение оператору и проверьте сырой ответ.', data.raw);
    } else showResult(data.output, input.policies, data.elapsed_seconds);
    text('runStatus', 'Разбор завершён. Проверьте содержание ответа по правилам компании.');
  } catch (error) {
    if (revision === requestRevision) showError(error.message);
    text('runStatus', 'Разбор не выполнен. Проверьте сообщение об ошибке.');
  } finally {
    busy = false;
    $('runBtn').disabled = $('retryBtn').disabled = false;
    $('loadingState').hidden = true;
    $('resultPane').setAttribute('aria-busy','false');
    if (revision !== requestRevision) $('emptyState').hidden = false;
  }
}

$('appeal').addEventListener('input', () => {
  $('factsJson').value = $('historyJson').value = '[]';
  text('exampleHint', 'Текст изменён: прежние факты и история сброшены. Добавьте актуальные сведения.');
  resetResult(true); contextPreview();
});
for (const id of ['rules','factsJson','historyJson']) $(id).addEventListener('input', () => { resetResult(true); contextPreview(); });
$('example').addEventListener('change', () => loadExample(Number($('example').value)));
$('runBtn').addEventListener('click', analyze);
$('retryBtn').addEventListener('click', analyze);
$('rawToggle').addEventListener('click', () => {
  $('rawOut').hidden = !$('rawOut').hidden;
  $('rawToggle').setAttribute('aria-expanded', String(!$('rawOut').hidden));
  text('rawToggle', $('rawOut').hidden ? 'Показать сырой ответ' : 'Скрыть сырой ответ');
});
for (const [id, format] of [['copyReply', output=>output.reply], ['copyJson', output=>JSON.stringify(output,null,2)]]) {
  $(id).addEventListener('click', async () => {
    if (!currentOutput) return;
    const span = $(id).querySelector('span');
    try { await navigator.clipboard.writeText(format(currentOutput)); span.textContent = 'Скопировано'; }
    catch { span.textContent = 'Не удалось скопировать'; }
    setTimeout(()=>{ span.textContent = 'Копировать'; }, 2000);
  });
}
fetch('/api/examples').then(r=>r.json()).then(data => {
  examples = data;
  $('example').replaceChildren(...data.map((example, index) => {
    const option = document.createElement('option'); option.value = index;
    const [category, state] = example.label.split(' / ');
    const states = {true:'условия выполнены',false:'условия не выполнены',unknown:'не хватает сведений',conflict:'конфликт правил',critical:'критический случай'};
    option.textContent = `${translated(category)} · ${states[state] || state}`;
    return option;
  }));
  const categories = ['refund','access','cancellation','technical'];
  $('exampleChips').replaceChildren(...categories.map(category => {
    const index = data.findIndex(example=>example.label.startsWith(category+' / '));
    const label = document.createElement('label'); label.className = 'chip';
    label.innerHTML = `<input type="radio" name="exampleChip" value="${index}"><span>${esc(translated(category))}</span>`;
    label.querySelector('input').addEventListener('change',()=>loadExample(index));
    return label;
  }));
  loadExample(0);
}).catch(()=>{ text('runStatus','Не удалось загрузить примеры. Обновите страницу.'); });
fetch('/api/status').then(r=>r.json()).then(data => {
  text('modelMode',data.mode==='LoRA'?'Локальная модель · LoRA':'Локальная модель · Base');
}).catch(()=>{ text('modelMode','Сервер недоступен'); });
