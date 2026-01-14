# Load packages
list.of.packages <- c("visNetwork", "reshape2", "RColorBrewer", "network", "scales", "htmlwidgets", "webshot")

# Install BiocManager and its libraries if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")}
bio.packages <- c("graph", "Rgraphviz")
bio.to.install <- bio.packages[!(bio.packages %in% installed.packages()[,"Package"])]
if (length(bio.to.install)) {
  BiocManager::install(bio.to.install, ask = FALSE)}

# Install missing libraries in CRAN
cran.packages <- setdiff(list.of.packages, bio.packages)
new.cran.packages <- cran.packages[!(cran.packages %in% installed.packages()[,"Package"])]
if (length(new.cran.packages)) {
  install.packages(new.cran.packages)
}
invisible(sapply(list.of.packages, require, character.only = TRUE))

# Install PhantomJS for webshot if not already installed
if (!webshot::is_phantomjs_installed()) {
  message("PhantomJS not found. Installing...")
  webshot::install_phantomjs()
}

library(bnstruct)
library(data.table)
library(visNetwork)


# Read the file into a data frame
data <- read.table("dbn_data.txt", header = FALSE, sep = "", stringsAsFactors = FALSE)
vars <- c('Sex', 'Age', 'OnsetSite', 'StudyArm', 'Riluzole', 'TimeSinceOnset',
          'Eosinophil_t1',	'Basophil_t1',	'Fvc_t1', 'Albumin_t1',	'AlkalinePhosphatase_t1',
          'ALT_t1',	'AST_t1',	'Bicarbonate_t1', 'Bilirubin_t1',	'BloodUreaNitrogen_t1',
          'Calcium_t1',	'Chloride_t1', 'CK_t1',	'Creatinine_t1', 'Glucose_t1',
          'Hematocrit_t1', 'Hemoglobin_t1', 'Platelets_t1', 'Potassium_t1', 'Protein_t1',
          'RBC_t1', 'Sodium_t1', 'WBC_t1',	'Gammaglutamyltransferase_t1', 'Diastolic_t1',
          'Systolic_t1', 'Pulse_t1', 'RespiratoryRate_t1',	'BMI_t1',
          'Q1Speech_t1', 'Q2Salivation_t1', 'Q3Swallowing_t1', 'Q4Handwriting_t1', 'Q5Cutting_t1',
          'Q6Dressing_t1', 'Q7Turning_t1', 'Q8Walking_t1', 'Q9Climbing_t1', 'Q10Respiratory_t1',
          'Eosinophil_t2', 'Basophil_t2',	'Fvc_t2',	'Albumin_t2', 'AlkalinePhosphatase_t2',
          'ALT_t2',	'AST_t2',	'Bicarbonate_t2', 'Bilirubin_t2', 'BloodUreaNitrogen_t2',
          'Calcium_t2', 'Chloride_t2', 'CK_t2', 'Creatinine_t2', 'Glucose_t2',
          'Hematocrit_t2', 'Hemoglobin_t2',	'Platelets_t2',	'Potassium_t2', 'Protein_t2',
          'RBC_t2', 'Sodium_t2', 'WBC_t2', 'Gammaglutamyltransferase_t2', 'Diastolic_t2',
          'Systolic_t2', 'Pulse_t2', 'RespiratoryRate_t2', 'BMI_t2',	
          'Q1Speech_t2', 'Q2Salivation_t2', 'Q3Swallowing_t2', 'Q4Handwriting_t2', 'Q5Cutting_t2',
          'Q6Dressing_t2', 'Q7Turning_t2', 'Q8Walking_t2', 'Q9Climbing_t2', 'Q10Respiratory_t2')

n_vars <- 84
disc <- rep("d", n_vars)
cardinalities <- c(2, 3, 4, 2, 2, 3,
                   2,	2, 3, 3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3,
                   5,	5, 5,	5, 5,
                   5,	5, 5,	5, 5,
                   2,	2, 3,	3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3, 3,
                   3,	3, 3,	3,
                   5,	5, 5,	5, 5,
                   5,	5, 5,	5, 5)
bndataset <- BNDataset(data=data,
                       discreteness=disc,
                       variables=vars,
                       node.sizes=cardinalities,
                       starts.from=0)
layer_vector <- c(1, 2, 2, 3, 3, 4,
                  5, 5,	5, 5,	5,
                  5, 5,	5, 5,	5,
                  5, 5, 5, 5, 5,
                  5, 5, 5, 5, 5,
                  5, 5, 5, 5, 5,
                  5, 5, 5, 5,
                  5, 5, 5, 5, 5,
                  5, 5, 5, 5, 5,
                  6, 6,	6, 6, 6,
                  6, 6,	6, 6, 6,
                  6, 6, 6, 6, 6,
                  6, 6,	6, 6, 6,
                  6, 6, 6, 6, 6,
                  6, 6, 6, 6,	
                  6, 6, 6, 6, 6,
                  6, 6, 6, 6, 6)

# Initialize matrix for mandatory.edges
mandatory.edges <- matrix(0, nrow = n_vars, ncol = n_vars)

# Set mandatory edges based on prior logic for Time Since Onset (TSO)
mandatory.edges[6, 46:84] <- 1

# Number of layers
n_layers <- 6

# Initialize matrix for layer.struct
layer.struct <- matrix(0, nrow = n_layers, ncol = n_layers)

layer.struct[1, 2] = 1; layer.struct[1, 3] = 1; layer.struct[1, 6] = 1; # 1 -> 2, 3, 6
layer.struct[2, 2] = 1; layer.struct[2, 3] = 1; layer.struct[2, 6] = 1; # 2 -> 2, 3, 6
layer.struct[3, 6] = 1; # 3 -> 6
layer.struct[4, 6] = 1; # 4 -> 6
layer.struct[5, 6] = 1; # 5 -> 6

layer_names <- c('Demographics', 'Onset', 'DiseaseTreatment',
                 'TimeSinceOnset', 'Dynamic_t1', 'Dynamic_t2')
rownames(layer.struct) <- layer_names
colnames(layer.struct) <- layer_names
net <- learn.network(bndataset,
                     layering=layer_vector, 
                     layer.struct=layer.struct,
                     mandatory.edges=mandatory.edges,
                     algo="hc",
                     scoring.func="BIC",
                     max.parents=5,
                     max.fanin=5)

plot(net)
saveRDS(net, file = "net.rds")

# Extract parameters
# Adjacency matrix
adj.mat <- as.matrix(net@dag)
vars <- net@variables
colnames(adj.mat) <- vars
rownames(adj.mat) <- vars
# Nodes
node.names <- colnames(adj.mat)
nodes <- data.frame(id = node.names, label = node.names)
# Edges
edges <- which(adj.mat == 1, arr.ind = TRUE)
edges.df <- data.frame(from = rownames(adj.mat)[edges[, "row"]],
                       to   = colnames(adj.mat)[edges[, "col"]])

# Basic network plot
visNetwork(nodes = nodes,
           edges = edges.df,
           height = "700px", width = "100%")

# Advanced network plot
visNet <- visNetwork(nodes, edges.df, height = "700px", width = "100%") %>%
  visNodes(shape = "oval", size = 40, font = list(size = 28, color = "black", face = "bold", align = "center")) %>%
  visEdges(arrows = "to", title = "function(edge) { return edge.title; }", color = list(highlight = "blue", inherit = FALSE)) %>%
  visPhysics(solver = "repulsion", repulsion = list(nodeDistance = 200, springLength = 200), stabilization = list(enabled = TRUE, iterations = 1000)) %>%
  visEvents(stabilizationIterationsDone = "function () {this.setOptions({physics:false});}") %>%
  visLayout(randomSeed = 456)
# Plot the graph
print(visNet)

# Save as html to maintain its interactivity
htmlwidgets::saveWidget(visNet, file = "net_original.html")


# Identify the target variables
target_vars <- c('Q1Speech_t2', 'Q2Salivation_t2', 'Q3Swallowing_t2', 'Q4Handwriting_t2', 'Q5Cutting_t2',
                 'Q6Dressing_t2', 'Q7Turning_t2', 'Q8Walking_t2', 'Q9Climbing_t2', 'Q10Respiratory_t2')

# Temperature scaling (soften/sharpen probabilities)
temp_scale <- function(p, temp = 1.2) {
  q <- p^(1/temp)
  q / sum(q)
}

extreme_skew <- function(probs, power = 1) {
  weights <- rev(seq_along(probs)) ^ power  # bias toward deterioration
  skewed_probs <- probs * weights
  skewed_probs <- skewed_probs / sum(skewed_probs)
}

# Variance inflation (add uncertainty)
variance_inflate <- function(p, gamma = 0.8) {
  # mix with a local symmetric kernel around the mode
  K <- length(p)
  i_mode <- which.max(p)
  kernel <- exp(-((1:K - i_mode)^2) / 2) # Gaussian-like
  kernel <- kernel / sum(kernel)
  q <- gamma * p + (1 - gamma) * kernel
  q / sum(q)
}

# ---- Generic CPT transformer wrapper ----
apply_CPT_transform <- function(net, target_vars, transform_fn, ...) {
  net_alt <- net
  for (var in target_vars) {
    cat("Processing", var, "\n")
    cpt <- net@cpts[[var]]
    
    # Dimensionality
    dims <- dim(cpt)
    n_dims <- length(dims)
    
    index_grid <- as.matrix(expand.grid(lapply(dims[-n_dims], seq_len)))
    
    for (i in seq_len(nrow(index_grid))) {
      idx <- as.list(index_grid[i,])
      probs <- do.call(`[`, c(list(cpt), idx, list(TRUE)))  # prob vector
      cpt_slice <- transform_fn(probs, ...)                 # apply custom fn
      cpt <- do.call(`[<-`, c(list(cpt), idx, list(TRUE), list(cpt_slice)))
    }
    net_alt@cpts[[var]] <- cpt
  }
  net_alt
}

predict_next_step <- function(net, vars, obs, next_cols=46:84) {
  dag <- net@dag
  for (child_idx in next_cols) {
    child <- vars[child_idx]
    parent_idx <- which(dag[, child_idx] == 1)
    
    # Get current values for the parent variables
    idx_chr <- as.character(unlist(obs[parent_idx]))
    
    # Construct and evaluate CPT access expression
    index_expr <- paste(c(idx_chr, ""), collapse=",")
    full_expr <- sprintf("net@cpts$%s[%s]", child, index_expr)
    prob_vec <- eval(parse(text=full_expr))
    
    # Normalize and sample new value
    prob_vec <- as.numeric(prob_vec) 
    prob_vec <- prob_vec / sum(prob_vec) # Probably not needed as this is already a probability distribution
    new_val <- sample(seq_along(prob_vec), size=1, prob = prob_vec)
    
    # Store prediction back into observation
    obs[[child_idx]] <- new_val
  }
  
  return(obs)
}

# Phase (TSO) distribution in the original data
phase_prob <- {
  tbl <- net@cpts$TimeSinceOnset # column 6 = TimeSinceOnset
  as.numeric(tbl) / sum(tbl)   # empirical proportions
}

# Helper: sample one row with the desired phase
sample_row_with_phase <- function(net, desired_phase) {
  repeat {
    obs <- sample.row(net)                 
    if (obs[6] == desired_phase)
      return(obs)
  }
}

save_df_as_tsv_file <- function(df, filename) {
  outfile <- file.path(getwd(), filename)
  write.table(df, file = outfile, sep = "\t", quote = FALSE, row.names = FALSE)
}

# Generate 1000 synthetic patients from a single phase
set.seed(123) # reproducibility
vars_out <- c("subject_id", vars[1:45])

count_patients_and_visits_per_phase <- function(df) {
  dt <- as.data.table(df)
  # Count visits per phase (Time Since Onset)
  visits_per_phase <- dt[, .N, by = TimeSinceOnset][order(TimeSinceOnset)]
  # Count unique patients per phase (Time Since Onset)
  patients_per_phase <- dt[, .(n_patients = uniqueN(subject_id)), by = TimeSinceOnset][order(TimeSinceOnset)]
  print(patients_per_phase)
  print(visits_per_phase)
}

cases <- list(
  list(n = 1000, path = c(1)),    # stays in 1
  list(n = 1000, path = c(2)),    # stays in 2
  list(n = 250, path = c(3)),     # stays in 3
  list(n = 2000, path = c(1, 2)), # 1 → 2
  list(n = 500, path = c(2, 3)),  # 2 → 3
  list(n = 50, path = c(1, 2, 3)) # 1 → 2 → 3
)

# Create a list of altered networks per phase
phase_temps <- list(
  "1" = 1,
  "2" = 1.2,
  "3" = 1.4
)

phase_powers <- list(
  "1" = 1,
  "2" = 2,
  "3" = 4
)

phase_gammas <- list(
  "1" = 1,
  "2" = 0.8,
  "3" = 0.6
)

phase_nets <- list()
phase_nets[["1"]] <- net
phase_nets[["2"]] <- net
phase_nets[["3"]] <- net

# Build a network per phase

# Build a network per phase
phase_nets_a <- list()
for (phase in names(phase_temps)) {
  temp <- phase_temps[[phase]]
  phase_nets_a[[phase]] <- apply_CPT_transform(net, target_vars, temp_scale, temp)
}

phase_nets_b <- list()
for (phase in names(phase_powers)) {
  power <- phase_powers[[phase]]
  phase_nets_b[[phase]] <- apply_CPT_transform(net, target_vars, extreme_skew, power)
}

phase_nets_c <- list()
for (phase in names(phase_gammas)) {
  gamma <- phase_gammas[[phase]]
  phase_nets_c[[phase]] <- apply_CPT_transform(net, target_vars, variance_inflate, gamma)
}

# Adding and removing edges for net for Phase 1
mandatory.edges <- matrix(0, nrow = n_vars, ncol = n_vars)
mandatory.edges[6, 46:84] <- 1
mandatory.edges[3, 75:84] <- 1
net1 <- learn.network(bndataset,
                      layering=layer_vector, 
                      layer.struct=layer.struct,
                      mandatory.edges=mandatory.edges,
                      algo="hc",
                      scoring.func="BIC",
                      max.parents=5,
                      max.fanin=5)

# Adding and removing edges for net for Phase 2
mandatory.edges <- matrix(0, nrow = n_vars, ncol = n_vars)
mandatory.edges[6, 46:84] <- 1
mandatory.edges[2, 75:84] <- 1
net2 <- learn.network(bndataset,
                      layering=layer_vector, 
                      layer.struct=layer.struct,
                      mandatory.edges=mandatory.edges,
                      algo="hc",
                      scoring.func="BIC",
                      max.parents=5,
                      max.fanin=5)

# Adding and removing edges for net for Phase 3
mandatory.edges <- matrix(0, nrow = n_vars, ncol = n_vars)
mandatory.edges[6, 46:84] <- 1
mandatory.edges[1, 75:84] <- 1
net3 <- learn.network(bndataset,
                      layering=layer_vector, 
                      layer.struct=layer.struct,
                      mandatory.edges=mandatory.edges,
                      algo="hc",
                      scoring.func="BIC",
                      max.parents=5,
                      max.fanin=5)

# Build a network per phase
phase_nets_d <- list()
phase_nets_d[["1"]] <- net1
phase_nets_d[["2"]] <- net2
phase_nets_d[["3"]] <- net3

# Variable groups for scenario E
# Q1, Q2, Q3, Q10 to get temp_scale + variance_inflate
vars_tempvar <- c('Q1Speech_t2', 'Q2Salivation_t2', 'Q3Swallowing_t2', 'Q10Respiratory_t2')
# Q4-Q9 to get extreme_skew + variance_inflate
vars_skewvar <- c('Q4Handwriting_t2', 'Q5Cutting_t2', 'Q6Dressing_t2',
                  'Q7Turning_t2', 'Q8Walking_t2', 'Q9Climbing_t2')
# Define power values for each phase
phase_powers_e <- list(
  "1" = 0,
  "2" = 2,
  "3" = 4
)
phase_temps_e <- list(
  "1" = 1,
  "2" = 1.2,
  "3" = 1.4
)
phase_gammas_e <- list(
  "1" = 0.7,
  "2" = 0.8,
  "3" = 0.9
)

# Build a network per phase for scenario E
phase_nets_e <- list()
for (phase in names(phase_powers)) {
  power <- phase_powers_e[[phase]]
  temp <- phase_temps_e[[phase]]
  gamma <- phase_gammas_e[[phase]]
  net_alt <- apply_CPT_transform(net, vars_skewvar, extreme_skew, power)
  net_alt <- apply_CPT_transform(net_alt, vars_tempvar, temp_scale, temp)
  net_alt <- apply_CPT_transform(net_alt, target_vars, variance_inflate, gamma)
  phase_nets_e[[phase]] <- net_alt
}

# Subpopulation temperature schedules
subpop_temps <- list(
  "A" = list("1" = 1.0, "2" = 1.2, "3" = 1.4),
  "B" = list("1" = 0.8, "2" = 1.2, "3" = 1.6),
  "C" = list("1" = 1.0, "2" = 1.4, "3" = 1.8)
)

# Build the three subpopulation phase_nets lists
phase_nets_f <- list()
for (subpop in names(subpop_temps)) {
  for (phase in names(subpop_temps[[subpop]])) {
    temp <- subpop_temps[[subpop]][[phase]]
    phase_nets_f[[subpop]][[phase]] <- apply_CPT_transform(net, target_vars, temp_scale, temp)
  }
}

cases_A <- list(
  list(n = 334, path = c(1)),    # stays in 1
  list(n = 333, path = c(2)),    # stays in 2
  list(n = 83, path = c(3)),     # stays in 3
  list(n = 667, path = c(1, 2)), # 1 → 2
  list(n = 167, path = c(2, 3)),  # 2 → 3
  list(n = 16, path = c(1, 2, 3)) # 1 → 2 → 3
)

cases_B <- list(
  list(n = 333, path = c(1)),    # stays in 1
  list(n = 334, path = c(2)),    # stays in 2
  list(n = 83, path = c(3)),     # stays in 3
  list(n = 666, path = c(1, 2)), # 1 → 2
  list(n = 167, path = c(2, 3)),  # 2 → 3
  list(n = 17, path = c(1, 2, 3)) # 1 → 2 → 3
)

cases_C <- list(
  list(n = 333, path = c(1)),    # stays in 1
  list(n = 333, path = c(2)),    # stays in 2
  list(n = 84, path = c(3)),     # stays in 3
  list(n = 667, path = c(1, 2)), # 1 → 2
  list(n = 166, path = c(2, 3)),  # 2 → 3
  list(n = 17, path = c(1, 2, 3)) # 1 → 2 → 3
)

unroll_patient_phases_per_phase <- function(phase_nets, vars, obs_vec,
                                            phase_seq,
                                            n_vis_seq,
                                            t1_idx = 7:45,
                                            t2_idx = 46:84) {
  stopifnot(length(phase_seq) == length(n_vis_seq))
  
  obs <- as.list(obs_vec)
  rows <- list()
  
  # Save first visit
  rows[[1]] <- unlist(obs[1:45], use.names = FALSE)
  
  # Shift t2 into t1 for second visit
  obs[t1_idx] <- obs[t2_idx]
  rows[[2]] <- unlist(obs[1:45], use.names = FALSE)
  
  for (seg in seq_along(phase_seq)) {
    phase <- phase_seq[seg]
    obs[[6]] <- phase
    
    net_curr <- phase_nets[[as.character(phase)]]
    stopifnot(!is.null(net_curr))
    
    n_vis <- n_vis_seq[seg]
    if (seg == 1) {
      n_vis <- n_vis - 2
    }
    
    for (v in seq_len(n_vis)) {
      obs <- predict_next_step(net_curr, vars, obs, next_cols = t2_idx)
      obs[t1_idx] <- obs[t2_idx]
      rows[[length(rows) + 1]] <- unlist(obs[1:45], use.names = FALSE)
    }
  }
  
  do.call(rbind, rows)
}

gen_patient_per_phase <- function(phase_nets, vars, phase_path, subj_id) {
  n_vis_seq <- pmax(5, round(rnorm(length(phase_path), 10, 2)))
  first_phase <- as.character(phase_path[1])
  net_for_first_phase <- phase_nets[[first_phase]]
  obs <- sample_row_with_phase(net_for_first_phase, phase_path[1])
  mat <- unroll_patient_phases_per_phase(phase_nets, vars, obs, phase_path, n_vis_seq)
  cbind(subject_id = subj_id, mat)
}

generate_multi_phase_patients <- function(phase_nets, n_patients, cases, vars_out, start_index = 0) {
  all_rows <- vector("list", n_patients)
  subj_id <- 1
  
  for (case in cases) {
    for (i in seq_len(case$n)) {
      print(start_index + subj_id)
      all_rows[[subj_id]] <- gen_patient_per_phase(phase_nets, vars, case$path, start_index + subj_id)
      subj_id <- subj_id + 1
    }
  }
  
  df <- as.data.frame(do.call(rbind, all_rows))
  colnames(df) <- vars_out
  return(df)
}

df <- generate_multi_phase_patients(phase_nets, 4800, cases, vars_out)
save_df_as_tsv_file(df, "multi_phase_scenario.tsv")
  
df_a <- generate_multi_phase_patients(phase_nets_a, 4800, cases, vars_out)
save_df_as_tsv_file(df_a, "multi_phase_scenario_a.tsv")
  
df_b <- generate_multi_phase_patients(phase_nets_b, 4800, cases, vars_out)
save_df_as_tsv_file(df_b, "multi_phase_scenario_b.tsv")
  
df_c <- generate_multi_phase_patients(phase_nets_c, 4800, cases, vars_out)
save_df_as_tsv_file(df_c, "multi_phase_scenario_c.tsv")
  
df_d <- generate_multi_phase_patients(phase_nets_d, 4800, cases, vars_out)
save_df_as_tsv_file(df_d, "multi_phase_scenario_d.tsv")
  
df_e <- generate_multi_phase_patients(phase_nets_e, 4800, cases, vars_out)
save_df_as_tsv_file(df_e, "multi_phase_scenario_e.tsv")

df_f_A <- generate_multi_phase_patients(phase_nets_f[["A"]], 1600, cases_A, vars_out)
df_f_B <- generate_multi_phase_patients(phase_nets_f[["B"]], 1600, cases_B, vars_out, 1600)
df_f_C <- generate_multi_phase_patients(phase_nets_f[["C"]], 1600, cases_C, vars_out, 3200)
df_f <- rbind(df_f_A, df_f_B, df_f_C)
save_df_as_tsv_file(df_f, "multi_phase_scenario_f.tsv")

generate_controlled_cohort_patient <- function(phase_nets, vars, phase_path, subj_id) {
  n_vis_seq <- c(7, 7, 7)
  first_phase <- as.character(phase_path[1])
  net_for_first_phase <- phase_nets[[first_phase]]
  obs <- sample_row_with_phase(net_for_first_phase, phase_path[1])
  mat <- unroll_patient_phases_per_phase(phase_nets, vars, obs, phase_path, n_vis_seq)
  cbind(subject_id = subj_id, mat)
}

generate_controlled_cohort_patients <- function(phase_nets, n_patients, vars_out, start_index = 0) {
  all_rows <- vector("list", n_patients)
  
  for (subj_id in seq_len(n_patients)) {
    print(start_index + subj_id)
    all_rows[[subj_id]] <- generate_controlled_cohort_patient(phase_nets, vars, c(1, 2, 3), start_index + subj_id)
  }
  
  df <- as.data.frame(do.call(rbind, all_rows))
  colnames(df) <- vars_out
  return(df)
}

df <- generate_controlled_cohort_patients(phase_nets, 3000, vars_out)
save_df_as_tsv_file(df, "controlled_cohort_scenario.tsv")

df_a <- generate_controlled_cohort_patients(phase_nets_a, 3000, vars_out)
save_df_as_tsv_file(df_a, "controlled_cohort_scenario_a.tsv")

df_b <- generate_controlled_cohort_patients(phase_nets_b, 3000, vars_out)
save_df_as_tsv_file(df_b, "controlled_cohort_scenario_b.tsv")

df_c <- generate_controlled_cohort_patients(phase_nets_c, 3000, vars_out)
save_df_as_tsv_file(df_c, "controlled_cohort_scenario_c.tsv")

df_d <- generate_controlled_cohort_patients(phase_nets_d, 3000, vars_out)
save_df_as_tsv_file(df_d, "controlled_cohort_scenario_d.tsv")

df_e <- generate_controlled_cohort_patients(phase_nets_e, 3000, vars_out)
save_df_as_tsv_file(df_e, "controlled_cohort_scenario_e.tsv")

df_f_A <- generate_controlled_cohort_patients(phase_nets_f[["A"]], 1000, vars_out)
df_f_B <- generate_controlled_cohort_patients(phase_nets_f[["B"]], 1000, vars_out, 1000)
df_f_C <- generate_controlled_cohort_patients(phase_nets_f[["C"]], 1000, vars_out, 2000)
df_f <- rbind(df_f_A, df_f_B, df_f_C)
save_df_as_tsv_file(df_f, "controlled_cohort_scenario_f.tsv")