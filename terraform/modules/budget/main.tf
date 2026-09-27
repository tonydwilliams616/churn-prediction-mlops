# aws_budgets_budget is AWS's native cost-alerting resource - it doesn't
# stop anything from running, it just watches your account's spend and
# emails you when it crosses thresholds you define. Plain email-notification
# budgets like this one are free; only "action-enabled" budgets that
# automatically take a remediation action (e.g. stopping resources) carry a
# small per-day charge.
resource "aws_budgets_budget" "this" {
  name         = var.budget_name
  budget_type  = "COST"
  limit_amount = var.monthly_limit_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # dynamic "notification" generates one notification block per value in
  # alert_threshold_percentages, rather than us hand-writing three nearly
  # identical blocks - this is the Terraform equivalent of a for-loop over
  # a resource's nested configuration.
  dynamic "notification" {
    for_each = var.alert_threshold_percentages

    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
