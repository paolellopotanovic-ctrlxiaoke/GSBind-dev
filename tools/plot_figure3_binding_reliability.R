args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep('^--file=', args, value = TRUE)
tool_dir <- dirname(normalizePath(sub('^--file=', '', file_arg[[1]])))
root <- normalizePath(file.path(tool_dir, '..'))
data_dir <- file.path(root, 'results/figure3_binding_reliability_20261021')
figure_dir <- file.path(data_dir, 'figures')
dir.create(figure_dir, showWarnings = FALSE)
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)
colors <- c(GSBind = '#2369A8', PeSTo = '#D95B79', GPSFun = '#31968C', ScanNet = '#C99231')
model_names <- c(gaussian_direct128 = 'GSBind', pesto_official = 'PeSTo', gpsfun_gpsite_pdb = 'GPSFun', scannet_ppi_no_msa = 'ScanNet')
read_data <- function(name) {
  table <- read.csv(file.path(data_dir, paste0(name, '.csv')), stringsAsFactors = FALSE)
  if ('model' %in% names(table)) table$model <- factor(unname(model_names[table$model]), levels = names(colors))
  if ('input_state' %in% names(table)) table$input_state <- factor(table$input_state, levels = c('apo', 'af2'), labels = c('Apo', 'AF2'))
  table
}
theme_local <- function() theme_classic(base_size = 8) + theme(
  text = element_text(family = 'sans'), plot.title = element_text(size = 9, face = 'bold'),
  plot.subtitle = element_text(size = 6.5, color = '#4A5560', margin = margin(b = 5)),
  axis.text = element_text(size = 7, color = 'black'), legend.position = 'top', legend.title = element_blank(),
  legend.text = element_text(size = 6), legend.key.size = unit(2.5, 'mm'),
  strip.background = element_rect(fill = '#F0F4F7', color = NA), strip.text = element_text(size = 8), plot.margin = margin(5, 6, 5, 5))
save_plot <- function(plot, name, width = 90, height = 75) {
  for (extension in c('pdf', 'png')) {
    ggsave(file.path(figure_dir, paste0(name, '.', extension)), plot, width = width, height = height,
           units = 'mm', dpi = 300, bg = 'white', limitsize = FALSE,
           device = if (extension == 'pdf') cairo_pdf else 'png')
  }
}
pair <- read_data('A_pair_and_H_packing_summary')
contrast <- read_data('A_pair_and_H_packing_paired_contrasts')
shift <- read_data('B_shift_summary')
stability <- read_data('C_stability_summary')
budget <- read_data('D_budget_summary')
region <- read_data('F_region_summary')
confidence <- read_data('G_common_confidence_summary')
cases <- read_data('pair_model_metrics')
system_metrics <- read_data('A_H_system_distributions')
system_stability <- read_data('C_system_distributions')
budget_grid <- read_data('D_budget_distribution_grid')
recall_curve <- read_data('E_full_recall_cost_summary')
confidence_systems <- read_data('G_system_distributions')
joint <- read_data('I_joint_system_metrics')
joint_summary <- read_data('I_joint_summary')
paired_delta <- cases %>% filter(model %in% c('GSBind', 'PeSTo')) %>%
  select(case_id, input_state, interface_ca_shift, model, state_ap) %>%
  pivot_wider(names_from = model, values_from = state_ap) %>% mutate(delta = GSBind - PeSTo)
shift_contrast <- read_data('B_shift_paired_contrasts') %>% filter(baseline == 'pesto_official', metric == 'state_ap')
shift_locations <- read_data('B_shift_case_metrics') %>% filter(model == 'GSBind') %>%
  group_by(input_state, shift_bin) %>% summarise(shift_location = median(interface_ca_shift), .groups = 'drop')
shift_contrast <- left_join(shift_contrast, shift_locations, by = c('input_state', 'shift_bin'))
model_axis <- scale_x_discrete(labels = c(GSBind = 'GS', PeSTo = 'PeSTo', GPSFun = 'GPS', ScanNet = 'Scan'))
distribution_plot <- function(table, summary, title, subtitle, axis_label) {
  ggplot(table, aes(model, value, color = model, fill = model)) +
    geom_violin(alpha = .12, linewidth = .3, trim = TRUE, scale = 'width') +
    geom_point(position = position_jitter(width = .13, height = 0, seed = 20261021), size = .35, alpha = .18) +
    geom_boxplot(width = .16, outlier.shape = NA, alpha = .45, linewidth = .35) +
    geom_errorbar(data = summary,
      inherit.aes = FALSE, mapping = aes(x = model, y = estimate, ymin = lower, ymax = upper),
      width = .18, linewidth = .5, color = '#202830') +
    geom_point(data = summary, aes(model, estimate), inherit.aes = FALSE, shape = 23,
      size = 1.5, fill = 'white', color = '#202830', stroke = .5) +
    scale_color_manual(values = colors) + scale_fill_manual(values = colors) + model_axis +
    labs(title = title, subtitle = subtitle, x = 'Prediction model', y = axis_label) + theme_local()
}
ap <- system_metrics %>% select(input_state, system_id, model, state_ap, ap_change) %>%
  pivot_longer(c(state_ap, ap_change), names_to = 'metric', values_to = 'value')
ap_summary <- pair %>% filter(metric %in% c('state_ap', 'ap_change'))
ap$metric <- factor(ap$metric, levels = c('state_ap', 'ap_change'), labels = c('Unbound AP', 'AP change vs holo'))
ap_summary$metric <- factor(ap_summary$metric, levels = c('state_ap', 'ap_change'), labels = c('Unbound AP', 'AP change vs holo'))
pa <- distribution_plot(ap, ap_summary, 'Accuracy and paired conformational change',
  'Dots: systems; box: IQR; white diamond: mean and 95% CI', 'System-level AP / paired AP change') +
  geom_hline(data = data.frame(metric = factor('AP change vs holo', levels = levels(ap$metric))),
    aes(yintercept = 0), color = '#999999', linewidth = .3, linetype = 2) +
  facet_grid(metric ~ input_state, scales = 'free_y')
pb <- ggplot(paired_delta, aes(interface_ca_shift, delta)) +
  geom_vline(xintercept = c(1, 2), linewidth = .25, color = '#bbbbbb', linetype = 3) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = .3) +
  geom_point(size = .65, alpha = .3, color = '#447F9F') +
  geom_rug(alpha = .15, linewidth = .2, sides = 'b') +
  geom_errorbar(data = shift_contrast, aes(x = shift_location, y = estimate, ymin = lower, ymax = upper),
                inherit.aes = FALSE, width = .05, linewidth = .65, color = '#b23b50') +
  geom_point(data = shift_contrast, aes(x = shift_location, y = estimate), inherit.aes = FALSE, size = 1.6, color = '#b23b50') +
  scale_x_continuous(trans = scales::pseudo_log_trans(base = 10), breaks = c(0, 1, 2, 5, 10, 20, 50)) +
  facet_wrap(~input_state) + labs(title = 'Accuracy across interface deformation', subtitle = 'All cases; red: system estimates in fixed bins (95% CI)', x = 'Interface CA displacement (A)', y = 'AP: GSBind - PeSTo') + theme_local()
retention <- system_stability %>% filter(requested_k == 0) %>%
  pivot_longer(c(jaccard, retained_correct_per_budget), names_to = 'metric', values_to = 'value')
retention_summary <- stability %>% filter(requested_k == 0, metric %in% c('jaccard', 'retained_correct_per_budget'))
retention$metric <- factor(retention$metric, levels = c('jaccard', 'retained_correct_per_budget'), labels = c('Candidate overlap', 'Shared correct / budget'))
retention_summary$metric <- factor(retention_summary$metric, levels = c('jaccard', 'retained_correct_per_budget'), labels = c('Candidate overlap', 'Shared correct / budget'))
pc <- distribution_plot(retention, retention_summary, 'Stable recommendations must be correct',
  'Top-10% budget: overlap versus retained true interface hits', 'System-level candidate agreement / correctness') +
  facet_grid(metric ~ input_state) + coord_cartesian(ylim = c(0, 1))
grid <- budget_grid %>% filter(requested_k > 0) %>%
  mutate(budget = factor(requested_k, levels = c(5, 10, 20, 40)),
    label = sprintf('%.0f%%\n[%.0f, %.0f]', mean * 100, q25 * 100, q75 * 100))
pd <- ggplot(grid, aes(budget, model, fill = mean)) + geom_tile(color = 'white', linewidth = .7) +
  geom_text(aes(label = label, color = mean > .65), size = 2.1, lineheight = .95) +
  scale_color_manual(values = c('FALSE' = '#1B2C3B', 'TRUE' = 'white'), guide = 'none') +
  scale_fill_gradient(low = '#F3F7FB', high = '#125990', limits = c(0, 1), name = 'Precision',
    breaks = c(0, .5, 1), labels = c('0%', '50%', '100%'),
    guide = guide_colorbar(barwidth = unit(22, 'mm'), barheight = unit(2, 'mm'))) +
  scale_y_discrete(limits = rev(names(colors))) + facet_wrap(~input_state) +
  labs(title = 'Fixed-budget yield and heterogeneity', subtitle = 'Each cell: system-mean precision; [system IQR, %]',
    x = 'Top K residues inspected', y = 'Prediction model') + theme_local()
pe <- ggplot(recall_curve, aes(target_recall, estimate, color = model, group = model)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = model), alpha = .08, color = NA, show.legend = FALSE) +
  geom_line(linewidth = .55) + geom_point(size = .7) +
  geom_vline(xintercept = c(.5, .8), linewidth = .25, color = '#999999', linetype = 3) +
  facet_wrap(~input_state) + scale_color_manual(values = colors) + scale_fill_manual(values = colors) +
  scale_x_continuous(breaks = c(.1, .5, .9), labels = scales::percent_format(accuracy = 1)) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format(accuracy = 1)) +
  labs(title = 'Less screening for interface coverage',
    subtitle = '80% recall: AF2 9.4%; Apo 16.6% less than PeSTo', x = 'Target recall (retrospective)', y = 'Fraction inspected (lower is better)') + theme_local()
region_wide <- region %>% filter(metric %in% c('interface_recall', 'inspected_fraction')) %>%
  select(input_state, requested_regions, model, metric, estimate, lower, upper) %>%
  pivot_wider(names_from = metric, values_from = c(estimate, lower, upper)) %>%
  rename(inspected_fraction = estimate_inspected_fraction, interface_recall = estimate_interface_recall)
pf_frontier <- ggplot(region_wide, aes(inspected_fraction, interface_recall, color = model, group = model)) +
  geom_segment(aes(x = lower_inspected_fraction, xend = upper_inspected_fraction, yend = interface_recall), linewidth = .25, alpha = .5) +
  geom_errorbar(aes(ymin = lower_interface_recall, ymax = upper_interface_recall), width = .001, linewidth = .25, alpha = .5) +
  geom_line(linewidth = .5) + geom_point(size = 1.5) +
  geom_text(aes(label = requested_regions), size = 2.1, vjust = -.6, show.legend = FALSE) +
  facet_wrap(~input_state) + scale_color_manual(values = colors) +
  labs(title = 'Localize the interface in spatial regions',
    subtitle = 'AF2, 3 regions: 37.8% vs 33.8% recall at ~13.6% area', x = 'Actual residue fraction inspected', y = 'Interface recall (1 / 3 / 5 regions)') + theme_local()
pf <- ggplot(region_wide, aes(factor(requested_regions), model)) +
  geom_point(aes(size = inspected_fraction, fill = interface_recall), shape = 21, color = '#687B8B', stroke = .3) +
  geom_text(aes(label = sprintf('%.0f', interface_recall * 100)), size = 2.1, color = '#172C3C') +
  scale_fill_gradient(low = '#EFF5FB', high = '#448BC1', limits = c(0, .6), name = 'Interface recall',
    breaks = c(0, .3, .6), labels = c('0%', '30%', '60%'),
    guide = guide_colorbar(barwidth = unit(18, 'mm'), barheight = unit(2, 'mm'))) +
  scale_size_area(max_size = 10, limits = c(0, .3), breaks = c(.05, .15, .25),
    labels = c('5%', '15%', '25%'), name = 'Residues inspected') +
  scale_y_discrete(limits = rev(names(colors))) + facet_wrap(~input_state) +
  labs(title = 'Spatial localization versus search area', subtitle = 'Number: recall (%); bubble area: actual residue fraction',
    x = 'Top nonredundant spatial regions', y = 'Prediction model') + theme_local()
confidence_systems$confidence_bin <- factor(confidence_systems$confidence_bin, levels = c('<70', '70-90', '>=90'))
dodge <- position_dodge(width = .8)
pg <- ggplot(confidence_systems, aes(confidence_bin, precision, color = model, fill = model)) +
  geom_point(position = position_jitterdodge(jitter.width = .13, jitter.height = 0, dodge.width = .8, seed = 20261021),
    size = .35, alpha = .15) +
  geom_boxplot(position = dodge, width = .65, outlier.shape = NA, alpha = .2, linewidth = .3) +
  geom_errorbar(data = filter(confidence, metric == 'precision'),
    aes(y = estimate, ymin = lower, ymax = upper), position = dodge, width = .1, linewidth = .45) +
  geom_point(data = filter(confidence, metric == 'precision'), aes(y = estimate), position = dodge, shape = 23, size = 1.2, fill = 'white') +
  scale_x_discrete(limits = c('<70', '70-90', '>=90'), labels = c('<70\n21 cases', '70-90\n170 cases', '>=90\n527 cases')) + scale_color_manual(values = colors) + scale_fill_manual(values = colors) +
  coord_cartesian(ylim = c(0, 1)) + labs(title = 'Structure confidence contextualizes hits',
    subtitle = 'Common case groups; confidence is not binding probability', x = 'GSBind candidate mean pLDDT', y = 'Top-10% interface precision') + theme_local()
packing_values <- system_metrics %>% transmute(input_state, system_id, model, value = packing_conditional_auc)
ph <- distribution_plot(packing_values, filter(pair, metric == 'packing_conditional_auc'),
  'Binding selectivity at similar packing', 'Compare only positive-negative pairs within packing strata',
  'System-level packing-conditional AUROC') + facet_wrap(~input_state) + coord_cartesian(ylim = c(0, 1))
joint$baseline <- factor(unname(model_names[joint$baseline]), levels = c('PeSTo', 'GPSFun', 'ScanNet'))
joint_summary$baseline <- factor(unname(model_names[joint_summary$baseline]), levels = c('PeSTo', 'GPSFun', 'ScanNet'))
joint_labels <- joint_summary %>% filter(quadrant == 'both_better') %>%
  mutate(label = sprintf('%d/%d (%.1f%%)', count, systems, estimate * 100))
pi <- ggplot(joint, aes(ap_gain, cost_saving)) +
  annotate('rect', xmin = 0, xmax = Inf, ymin = 0, ymax = Inf, fill = '#EAF3FA', alpha = .8) +
  geom_hline(yintercept = 0, linewidth = .3, color = '#777777') +
  geom_vline(xintercept = 0, linewidth = .3, color = '#777777') +
  geom_point(color = '#A0A8AF', alpha = .4, size = .55) +
  geom_point(data = filter(joint, quadrant == 'both_better'), color = colors[['GSBind']], alpha = .65, size = .65) +
  geom_text(data = joint_labels, aes(label = label), x = Inf, y = Inf, hjust = 1.05, vjust = 1.4,
    inherit.aes = FALSE, size = 2.1, color = colors[['GSBind']]) +
  facet_grid(input_state ~ baseline) + coord_cartesian(xlim = c(-1, 1), ylim = c(-1, 1)) +
  scale_x_continuous(breaks = c(-1, 0, 1)) + scale_y_continuous(breaks = c(-1, 0, 1)) +
  labs(title = 'Accuracy and screening gains coincide',
    subtitle = 'Blue: both better; one dot per system; all systems retained',
    x = 'AP gain: GSBind - comparator', y = '80%-recall cost saved: comparator - GSBind') + theme_local()
plots <- list(pa, pb, pc, pd, pe, pf, pg, ph, pi)
names(plots) <- c('A_transfer', 'B_deformation', 'C_correct_retention', 'D_fixed_budget', 'E_coverage_cost', 'F_regions', 'G_confidence', 'H_packing', 'I_joint_benefit')
for (name in names(plots)) save_plot(plots[[name]], name)
save_plot(pf_frontier, 'F_region_efficiency_frontier')
figure <- ggarrange(plotlist = plots, ncol = 3, nrow = 3, labels = letters[1:9],
                    font.label = list(size = 12, face = 'bold'), common.legend = FALSE)
save_plot(figure, 'GSBind_figure3_binding_v20261021', width = 330, height = 300)
main <- ggarrange(pa, pc, pd, pe, pf, ph, ncol = 3, nrow = 2, labels = letters[1:6],
  common.legend = FALSE, font.label = list(size = 12, face = 'bold'))
save_plot(main, 'GSBind_figure3_binding_core6_v20261021', width = 330, height = 200)
