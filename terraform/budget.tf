# Calling a module looks like calling a resource, but "module" blocks
# instantiate everything inside modules/budget/ as a self-contained unit,
# passing in the values it asks for via its variables.tf. This is the
# payoff of the module structure: this whole file could be copy-pasted with
# different values to create a second, independent budget (e.g. one for a
# separate "staging" environment) without touching modules/budget/ at all.
module "monthly_budget" {
  source = "./modules/budget"

  budget_name                 = "${var.project_name}-monthly-budget"
  monthly_limit_usd           = "20"
  alert_email                 = var.alert_email
  alert_threshold_percentages = [50, 80, 100]
}
