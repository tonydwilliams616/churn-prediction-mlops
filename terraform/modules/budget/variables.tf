# A module's variables.tf defines its own private "settings" - the inputs
# whoever USES this module must (or can) supply. This is exactly what makes
# a module reusable: someone could call this same module twice, with two
# different budget_name/monthly_limit_usd values, to get two independent
# budgets - e.g. one per environment.

variable "budget_name" {
  description = "Name for this budget - must be unique within the account"
  type        = string
}

variable "monthly_limit_usd" {
  description = "Monthly spending limit in USD that triggers alerts"
  type        = string
}

variable "alert_email" {
  description = "Email address to notify when spending crosses a threshold"
  type        = string
}

variable "alert_threshold_percentages" {
  description = "List of percentages of the budget at which to send an alert (e.g. 50 = alert at 50% of monthly_limit_usd spent)"
  type        = list(number)
  default     = [50, 80, 100]
}
