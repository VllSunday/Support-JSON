"""Authored scenario families. Policy text and executable conditions are kept together."""


def leaf(key, op, value, origin=None):
    result = {"key": key, "op": op, "value": value}
    if origin:
        result["origin"] = origin
    return result


def spec(family, category, text, message, condition, yes, no, true_facts, false_facts,
         unknown_key, labels, unknown="ask_details", priorities=("medium", "medium")):
    return {"family": family, "category": category, "text": text, "message": message,
            "condition": condition, "true": yes, "false": no,
            "true_facts": true_facts, "false_facts": false_facts,
            "unknown_key": unknown_key, "fact_labels": labels,
            "unknown_action": unknown, "priority_true": priorities[0],
            "priority_false": priorities[1]}


EXPLAIN = ("answer", "explain_policy", "По переданным правилам условия этого запроса не выполнены. Предложить запрошенную операцию на этом основании нельзя.")
REVIEW = ("escalate", "human_review", "По правилам компании этот случай требует рассмотрения специалистом поддержки.")
REFUND = ("propose_action", "propose_refund", "Условия возврата соблюдены. Можно предложить заявку на возврат; сама операция пока не выполнена.")
CANCEL = ("propose_action", "propose_cancellation", "Условия отмены соблюдены. Можно предложить заявку на отмену подписки; подписка пока не изменена.")
RESET = ("propose_action", "propose_password_reset", "По правилам можно предложить сброс пароля. Пароль пока не изменён.")
DIAGNOSTICS = ("propose_action", "propose_diagnostics", "По правилам следующий шаг — диагностика проблемы. Подтверждённого срока устранения пока нет.")
INVOICE = ("propose_action", "propose_invoice_download", "Условия выполнены: можно предложить скачать счёт в разделе «Документы» личного кабинета.")
PLAN = ("propose_action", "propose_plan_change", "Условия соблюдены: можно предложить изменение тарифа. Тариф пока не изменён.")
INTEGRATION = ("propose_action", "propose_integration_check", "По правилам следующий шаг — проверка настроек интеграции. Изменения пока не выполнены.")


def scenarios(split, profile):
    days, cutoff, seats, retention = profile["refund_days"], profile["cancel_cutoff"], profile["seats"], profile["retention"]
    if split == "train":
        return [
            spec("refund_recent_charge", "refund", f"Возврат разрешён, если после списания прошло не больше {days} дней включительно. Иначе объяснить отказ. При неизвестном сроке уточнить его.",
                 "Хочу вернуть оплату за сервис.", leaf("days_since_charge", "le", days), REFUND, EXPLAIN,
                 {"days_since_charge": days}, {"days_since_charge": days + 1}, "days_since_charge", {"days_since_charge": "сколько дней прошло после списания"}),
            spec("cancel_before_renewal", "cancellation", f"Отмену можно предложить, если до продления осталось не меньше {cutoff} дней. При меньшем сроке требуется человек; неизвестный срок уточнить.",
                 "Мне нужно отменить подписку до продления.", leaf("days_until_renewal", "ge", cutoff), CANCEL, REVIEW,
                 {"days_until_renewal": cutoff}, {"days_until_renewal": cutoff - 1}, "days_until_renewal", {"days_until_renewal": "сколько дней осталось до продления"}),
            spec("password_account_lock", "access", "Если система подтверждает, что аккаунт не заблокирован, предложить сброс пароля. При блокировке или отсутствии подтверждения системы требуется человек.",
                 "Не получается войти по паролю.", leaf("account_locked", "eq", False, "tool_observation"), RESET, REVIEW,
                 {"account_locked": False}, {"account_locked": True}, "account_locked", {"account_locked": "статус блокировки"}, "escalate", ("high", "medium")),
            spec("service_mass_incident", "technical", "При подтверждённом системой массовом сбое требуется человек. Если система подтверждает отсутствие массового сбоя, предложить диагностику. При неизвестном статусе требуется человек. Срок исправления не обещать.",
                 "Сервис не открывается, нужна помощь.", leaf("mass_outage", "eq", True, "tool_observation"), REVIEW, DIAGNOSTICS,
                 {"mass_outage": True}, {"mass_outage": False}, "mass_outage", {"mass_outage": "подтверждение массового сбоя"}, "escalate", ("critical", "high")),
            spec("duplicate_confirmed_charge", "billing", "Повторное списание проверяется по подтверждению системы. Если оно подтверждено, требуется человек. Если система подтверждает отсутствие дубля, сообщить это; при неизвестном статусе также нужен человек.",
                 "Кажется, оплату списали дважды.", leaf("duplicate_confirmed", "eq", True, "tool_observation"), REVIEW,
                 ("answer", "explain_policy", "В доступных данных системы повторное списание не подтверждено."),
                 {"duplicate_confirmed": True}, {"duplicate_confirmed": False}, "duplicate_confirmed", {"duplicate_confirmed": "подтверждение повторного списания"}, "escalate", ("high", "medium")),
            spec("invoice_retention", "billing", f"Скачать счёт в разделе «Документы» можно, если с оплаты прошло не больше {retention} дней. При более старой оплате нужен человек. Неизвестный срок уточнить.",
                 "Нужен счёт по моей оплате.", leaf("payment_age_days", "le", retention), INVOICE, REVIEW,
                 {"payment_age_days": retention}, {"payment_age_days": retention + 1}, "payment_age_days", {"payment_age_days": "сколько дней прошло после оплаты"}, priorities=("low", "low")),
            spec("plan_seat_limit", "product", f"Предложить изменение на выбранный тариф можно, если требуется не больше {seats} рабочих мест. Иначе объяснить ограничение. Количество мест, если неизвестно, уточнить.",
                 "Хочу перейти на выбранный тариф для команды.", leaf("requested_seats", "le", seats), PLAN, EXPLAIN,
                 {"requested_seats": seats}, {"requested_seats": seats + 1}, "requested_seats", {"requested_seats": "сколько рабочих мест требуется"}, priorities=("low", "low")),
            spec("integration_active_connector", "integration", "Проверку настроек можно предложить только для включённой по данным системы интеграции. Для отключённой объяснить ограничение; при неизвестном статусе нужен человек.",
                 "Данные перестали приходить из интеграции.", leaf("connector_enabled", "eq", True, "tool_observation"), INTEGRATION, EXPLAIN,
                 {"connector_enabled": True}, {"connector_enabled": False}, "connector_enabled", {"connector_enabled": "статус включения интеграции"}, "escalate"),
        ]
    if split == "validation":
        return [
            spec("refund_card_and_age", "refund", f"Возврат разрешён только для оплаты картой и в пределах {days} дней после списания. Иначе объяснить отказ. Уточнять только сведения, от которых ещё зависит решение.",
                 "Подскажите условия возврата моей оплаты.", {"all": [leaf("days_since_charge", "le", days), leaf("payment_method", "eq", "card")]}, REFUND, EXPLAIN,
                 {"days_since_charge": days, "payment_method": "card"}, {"days_since_charge": days, "payment_method": "transfer"}, "payment_method", {"days_since_charge": "срок после списания", "payment_method": "способ оплаты"}),
            spec("cancel_balance_clear", "cancellation", f"Отмена разрешена, если нет задолженности и до продления минимум {cutoff} дней. Иначе нужен человек. Неизвестные сведения клиента уточнить.",
                 "Можно прекратить продление моего тарифа?", {"all": [leaf("balance_clear", "eq", True), leaf("days_until_renewal", "ge", cutoff)]}, CANCEL, REVIEW,
                 {"balance_clear": True, "days_until_renewal": cutoff}, {"balance_clear": False, "days_until_renewal": cutoff}, "balance_clear", {"balance_clear": "есть ли задолженность", "days_until_renewal": "срок до продления"}),
            spec("access_local_auth_type", "access", "При входе с локальным паролем можно предложить сброс пароля. При другом типе входа нужен человек. Если тип входа неизвестен, уточнить его.",
                 "Как восстановить доступ к профилю?", leaf("auth_type", "eq", "local"), RESET, REVIEW,
                 {"auth_type": "local"}, {"auth_type": "sso"}, "auth_type", {"auth_type": "тип входа: пароль или SSO"}, priorities=("high", "medium")),
            spec("integration_token_expired", "integration", "Если клиент сообщает об истёкшем токене, предложить проверку настроек интеграции. Для другого сообщения об ошибке нужен человек. Отсутствующее описание ошибки уточнить.",
                 "Что делать с ошибкой подключения внешнего сервиса?", leaf("error_code", "eq", "TOKEN_EXPIRED"), INTEGRATION, REVIEW,
                 {"error_code": "TOKEN_EXPIRED"}, {"error_code": "UNKNOWN_ERROR"}, "error_code", {"error_code": "текст или код ошибки"}),
            spec("technical_confirmed_interruption", "technical", "При подтверждённом системой общем перерыве обслуживания нужен человек; при подтверждённом отсутствии перерыва предложить диагностику. Если система не даёт подтверждения, нужен человек.",
                 "Работа сервиса внезапно прервалась.", leaf("service_interruption", "eq", True, "tool_observation"), REVIEW, DIAGNOSTICS,
                 {"service_interruption": True}, {"service_interruption": False}, "service_interruption", {"service_interruption": "подтверждение перерыва"}, "escalate", ("critical", "high")),
            spec("billing_payment_unrecognized", "billing", "Если система не распознаёт полученный платёж, нужен человек. Если платёж распознан, сообщить это. Без подтверждения системы также нужен человек.",
                 "Оплатил, но не понимаю, дошла ли оплата.", leaf("payment_unrecognized", "eq", True, "tool_observation"), REVIEW,
                 ("answer", "explain_policy", "По доступным данным системы платёж распознан."),
                 {"payment_unrecognized": True}, {"payment_unrecognized": False}, "payment_unrecognized", {"payment_unrecognized": "статус распознавания оплаты"}, "escalate", ("high", "medium")),
            spec("invoice_available_document", "billing", "Если система подтверждает наличие счёта, предложить скачать его в разделе «Документы». Если счёта нет или его наличие не подтверждено, нужен человек.",
                 "Как получить документ по оплате?", leaf("invoice_available", "eq", True, "tool_observation"), INVOICE, REVIEW,
                 {"invoice_available": True}, {"invoice_available": False}, "invoice_available", {"invoice_available": "наличие счёта"}, "escalate", ("low", "low")),
            spec("product_plan_region", "product", "Предложить изменение тарифа можно только для региона EU. Для другого региона объяснить ограничение; неизвестный регион уточнить.",
                 "Хочу сменить тариф, есть ли ограничения по региону?", leaf("account_region", "eq", "EU"), PLAN, EXPLAIN,
                 {"account_region": "EU"}, {"account_region": "APAC"}, "account_region", {"account_region": "регион аккаунта"}, priorities=("low", "low")),
        ]
    if split == "test":
        return [
            spec("refund_trial_exception", "refund", f"Возврат разрешён в пределах {days} дней после списания ИЛИ при действующем пробном периоде. Достаточно одного условия. Иначе отказать; необходимые неизвестные сведения уточнить.",
                 "Можно получить назад деньги за текущую подписку?", {"any": [leaf("days_since_charge", "le", days), leaf("trial_active", "eq", True)]}, REFUND, EXPLAIN,
                 {"days_since_charge": days + 5, "trial_active": True}, {"days_since_charge": days + 5, "trial_active": False}, "trial_active", {"days_since_charge": "сколько дней прошло после списания", "trial_active": "действует ли пробный период"}),
            spec("refund_monthly_only", "refund", f"Возврат возможен только при месячной подписке и не позже {days} дней после списания. Годовая подписка этим правилом не покрывается. Неизвестные нужные сведения уточнить.",
                 "Прошу рассмотреть возврат платы за тариф.", {"all": [leaf("subscription_term", "eq", "monthly"), leaf("days_since_charge", "le", days)]}, REFUND, EXPLAIN,
                 {"subscription_term": "monthly", "days_since_charge": days}, {"subscription_term": "annual", "days_since_charge": days}, "subscription_term", {"subscription_term": "месячная или годовая подписка", "days_since_charge": "срок после списания"}),
            spec("cancel_annual_notice", "cancellation", "Для годовой подписки отмену можно предложить при уведомлении минимум за 14 дней до продления. Если хотя бы одно условие не выполнено, нужен человек. Неизвестные необходимые сведения уточнить.",
                 "Хочу остановить автоматическое продление годового обслуживания.", {"all": [leaf("subscription_term", "eq", "annual"), leaf("days_until_renewal", "ge", 14)]}, CANCEL, REVIEW,
                 {"subscription_term": "annual", "days_until_renewal": 14}, {"subscription_term": "annual", "days_until_renewal": 13}, "days_until_renewal", {"subscription_term": "тип подписки", "days_until_renewal": "срок до продления"}),
            spec("access_sso_admin", "access", "Для SSO стандартный сброс пароля доступен только администратору организации. Администратору предложить сброс; другим ролям нужен человек. Неизвестную роль уточнить.",
                 "Не могу авторизоваться через корпоративный вход.", leaf("organization_role", "eq", "admin"), RESET, REVIEW,
                 {"organization_role": "admin"}, {"organization_role": "member"}, "organization_role", {"organization_role": "роль в организации"}, priorities=("high", "medium")),
            spec("technical_workspace_incident", "technical", "Если система подтверждает инцидент, затрагивающий весь workspace, нужен человек. При подтверждённом отсутствии такого инцидента предложить диагностику. При недоступном подтверждении нужен человек; время восстановления не обещать.",
                 "В рабочем пространстве пропали возможности работать с документами.", leaf("workspace_incident", "eq", True, "tool_observation"), REVIEW, DIAGNOSTICS,
                 {"workspace_incident": True}, {"workspace_incident": False}, "workspace_incident", {"workspace_incident": "подтверждение общего инцидента"}, "escalate", ("critical", "high")),
            spec("billing_user_vs_system", "billing", "Рассматривать повторное списание как подтверждённое можно только по наблюдению инструмента. Подтверждённый дубль требует человека; подтверждённое отсутствие дубля можно сообщить. Утверждение клиента не заменяет наблюдение инструмента; при нехватке подтверждения нужен человек.",
                 "У меня отображаются две оплаты за один период, разберитесь, пожалуйста.", leaf("duplicate_observed", "eq", True, "tool_observation"), REVIEW,
                 ("answer", "explain_policy", "По доступному подтверждению системы повторное списание не обнаружено."),
                 {"duplicate_observed": True}, {"duplicate_observed": False}, "duplicate_observed", {"duplicate_observed": "подтверждение дубля"}, "escalate", ("high", "medium")),
            spec("invoice_verified_details", "billing", "Скачать счёт в разделе «Документы» можно только после подтверждения системой реквизитов организации. Если реквизиты не подтверждены либо их статус неизвестен, нужен человек.",
                 "Нужно получить счёт с реквизитами нашей организации.", leaf("company_details_verified", "eq", True, "tool_observation"), INVOICE, REVIEW,
                 {"company_details_verified": True}, {"company_details_verified": False}, "company_details_verified", {"company_details_verified": "подтверждение реквизитов"}, "escalate", ("low", "low")),
            spec("product_audit_enterprise", "product", "Переход ради функции аудита можно предложить только на тариф Enterprise. При выборе другого тарифа объяснить ограничение; если название тарифа неизвестно, уточнить его.",
                 "Хочу подключить тариф, чтобы смотреть журнал аудита.", leaf("target_plan", "eq", "enterprise"), PLAN, EXPLAIN,
                 {"target_plan": "enterprise"}, {"target_plan": "basic"}, "target_plan", {"target_plan": "какой тариф вы хотите подключить"}, priorities=("low", "low")),
            spec("integration_allowed_connector", "integration", "Проверку интеграции можно предложить только для коннекторов mail и crm. Для остальных объяснить, что правило их не поддерживает. Если имя коннектора неизвестно, уточнить его.",
                 "Не синхронизируется подключённый внешний инструмент.", leaf("connector_name", "in", ["mail", "crm"]), INTEGRATION, EXPLAIN,
                 {"connector_name": "crm"}, {"connector_name": "custom"}, "connector_name", {"connector_name": "название коннектора"}),
        ]
    raise ValueError(f"Unknown split: {split}")
