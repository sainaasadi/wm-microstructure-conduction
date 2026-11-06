# =============================================================================
# White matter microstructure and conduction velocity: edge-wise analysis
#
# Inputs  : group-level connectivity matrices (CSV, no header), one value per
#           region pair, regions ordered as in region_order.csv
#             - SMI metrics: deperp, f, p2
#             - axon radius (axond) and qT1 (t1); R1 (r1) for the g-ratio test only
#             - fibre length (len) and F-TRACT DCM conduction delay (delay)
# Outputs : edge tables (CSV) and figures, written to `out_dir`
#
# Run interactively (e.g. in RStudio) to see model summaries and plots.
# =============================================================================

## ---- Packages ---------------------------------------------------------------
library(ggplot2)
library(ggpubr)      # stat_cor
library(GGally)      # ggpairs
library(psych)       # corr.test
library(corrplot)
library(ggfortify)   # autoplot() for lm diagnostics
library(car)         # vif
library(effectsize)  # standardize_parameters
library(lm.beta)
library(dplyr)

## ---- Paths (edit to your own) -----------------------------------------------
data_dir <- "path/to/matrices"   # connectivity matrices and region labels (Zenodo)
out_dir  <- "path/to/output"     # edge tables and figures

## ---- Helper functions -------------------------------------------------------
# Drop rows with any value more than `threshold` SD from its column mean (NAs ignored)
remove_outliers_sd <- function(data, threshold = 3) {
  mean_vals <- apply(data, 2, mean, na.rm = TRUE)
  sd_vals   <- apply(data, 2, sd,   na.rm = TRUE)
  lower <- mean_vals - threshold * sd_vals
  upper <- mean_vals + threshold * sd_vals
  keep <- apply(data, 1, function(row) {
    ok <- !is.na(row)
    all(row[ok] >= lower[ok] & row[ok] <= upper[ok])
  })
  data[keep, ]
}

# Upper triangle of a matrix (diagonal excluded), in column-major order
get_upper_tri <- function(m) {
  m <- as.matrix(m)
  m[upper.tri(m)]
}

# "ctx-lh-A-ctx-rh-B" -> "intra" if both regions are in the same hemisphere
edge_hemi_type <- function(edge) {
  parts <- strsplit(edge, "-")[[1]]
  if (length(parts) >= 6 && parts[2] == parts[5]) "intra" else "inter"
}

## ---- Load matrices ----------------------------------------------------------
regions <- read.csv(file.path(data_dir, "region_order.csv"), header = FALSE)$V1

read_mat <- function(dir, file) {
  m <- as.matrix(read.csv(file.path(dir, file), header = FALSE))
  dimnames(m) <- list(regions, regions)
  m
}

mats <- list(
  deperp = read_mat(data_dir,   "connmat_deperp.csv"),
  f      = read_mat(data_dir,   "connmat_f.csv"),
  p2     = read_mat(data_dir,   "connmat_p2.csv"),
  axond  = read_mat(data_dir, "connmat_axond.csv"),
  t1     = read_mat(data_dir,    "connmat_myelin.csv"),     # qT1
  r1     = read_mat(data_dir,    "connmat_myelin_r1.csv"),  # R1, g-ratio test only
  len    = read_mat(data_dir,   "connmat_length.csv"),
  delay  = read_mat(data_dir,   "conn_delays_dcm_masked_by_counts_thr_10.csv") # Available from F-TRACT
)
mats$vel <- mats$len / mats$delay   # conduction velocity (m/s)

## ---- Cortico-cortical edges -------------------------------------------------
is_ctx <- grepl("ctx-", regions)
ctx    <- regions[is_ctx]
mats   <- lapply(mats, function(m) m[is_ctx, is_ctx])

# Velocity and delay are directed: average both directions (NA if either is missing)
for (v in c("vel", "delay")) mats[[v]] <- (mats[[v]] + t(mats[[v]])) / 2

vars   <- c("deperp", "f", "p2", "axond", "t1", "vel", "len", "delay")
df_all <- as.data.frame(sapply(mats[vars], get_upper_tri))
rownames(df_all) <- unlist(lapply(2:length(ctx), function(j) paste(ctx[1:(j - 1)], ctx[j], sep = "-")))

## ---- Filtering --------------------------------------------------------------
edges <- df_all[rowSums(df_all == 0, na.rm = TRUE) == 0, ]           # no zero in any metric
edges <- remove_outliers_sd(edges)                                     # within 3 SD
edges <- edges[is.na(edges$vel) | (edges$vel >= 0 & edges$vel <= 80), ]  # drop implausible velocities

edges$edge_label <- rownames(edges)
edges$edge_type  <- factor(sapply(edges$edge_label, edge_hemi_type), levels = c("intra", "inter"))

# Edge tables used by 02_velocity-prediction_figures.py
write.csv(cbind(edges = rownames(edges), edges),
          file.path(out_dir, "filtered_micro_delay_dcm10thr_ctx.csv"), row.names = FALSE)
write.csv(cbind(edges = rownames(df_all), df_all),
          file.path(out_dir, "all_micro_delay_dcm10thr_ctx.csv"), row.names = FALSE)

## ---- Correlations -----------------------------------------------------------
res_cor <- corr.test(as.matrix(edges[vars]), method = "spearman", adjust = "fdr")
corrplot(res_cor$r, type = "upper", p.mat = res_cor$p, sig.level = 0.05, insig = "blank",
         method = "color", addCoef.col = "black", number.cex = 0.7, tl.cex = 0.5, diag = FALSE,
         col = rev(colorRampPalette(c("darkred", "white", "darkblue"))(200)),
         title = "Correlation All (fdr corrected)", mar = c(1, 0, 1, 0))

# Axon radius distribution across cortico-cortical edges 
mean(df_all$axond, na.rm = TRUE)
sd(df_all$axond, na.rm = TRUE)
axond_plot <- df_all$axond[df_all$axond >= 0.4]
p_density <- ggplot(data.frame(axond = axond_plot), aes(x = axond)) +
  geom_density(fill = "darkblue", alpha = 0.7) +
  geom_vline(xintercept = mean(axond_plot), color = "darkred", linetype = "dashed", linewidth = 1) +
  labs(title = "Density of axon radius", x = "Axon radius (µm)", y = "Density") +
  theme_minimal()
p_density
ggsave(file.path(out_dir, "axond_density.pdf"), p_density)

# Pairwise Spearman correlations and scatterplots
cor_spearman <- function(size) {
  function(data, mapping, ...) ggally_cor(data = data, mapping = mapping, size = size, method = "spearman", ...)
}
scatter_all <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    geom_point(alpha = 0.5, size = 0.5, color = "black") +
    theme_classic() +
    geom_smooth(method = "lm", color = "darkred")
}
scatter_group <- function(data, mapping, ...) {
  ggplot(data = data, mapping = mapping) +
    geom_point(alpha = 0.5, size = 0.5) +
    theme_classic() +
    geom_smooth(method = "lm")
}

# All edges --> Figure 5
ggpairs(edges[vars],
        upper = list(continuous = cor_spearman(5)),
        lower = list(continuous = scatter_all),
        diag  = list(continuous = wrap("densityDiag", color = "darkred")),
        title = "Spearman Correlation and Scatterplots for Filtered Data") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Intra- vs inter-hemispheric edges --> Supplementary Figure 2
ggpairs(edges, columns = 1:8,
        mapping = aes(color = edge_type, fill = edge_type),
        upper = list(continuous = cor_spearman(3)),
        lower = list(continuous = scatter_group),
        diag  = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Spearman Correlation and Scatterplots for Filtered Data") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## ---- Main model: microstructure -> velocity ---------------------------------
m_main <- lm(log(vel) ~ axond + f + deperp + t1, data = edges)  # p2 not a significant predictor
summary(m_main)
autoplot(m_main, which = 1:4, label.n = 0)   # residuals, Q-Q, scale-location, Cook's distance
vif(m_main, type = "predictor")              # multicollinearity
standardize_parameters(m_main, method = "refit")
summary(lm.beta(m_main))

## ---- Intra- vs inter-hemispheric connections --------------------------------
m_t1_type    <- lm(log(vel) ~ axond + f + deperp + t1 * edge_type, data = edges)
m_type       <- lm(log(vel) ~ axond + f + deperp + t1 + edge_type, data = edges)
m_axond_type <- lm(log(vel) ~ axond * edge_type + f + deperp + t1, data = edges)
summary(m_t1_type)
summary(m_type)
summary(m_axond_type)

m_intra <- lm(log(vel) ~ axond + f + deperp, data = subset(edges, edge_type == "intra"))
m_inter <- lm(log(vel) ~ f + t1,             data = subset(edges, edge_type == "inter"))
summary(m_intra)
summary(m_inter)

ggplot(edges, aes(x = axond, y = log(vel), color = edge_type)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
  labs(x = "Axon Radius (µm)", y = "Observed log(Velocity)", color = "Connection Type") +
  theme_minimal() +
  scale_color_manual(values = c("intra" = "#E69F00", "inter" = "#56B4E9"))

## ---- g-ratio proxy: axon radius x R1 interaction (Supplementary Table 1) ----
# R1 is attached to the filtered edges here, so it does not affect which edges are kept
r1_all   <- setNames(get_upper_tri(mats$r1), rownames(df_all))
edges$r1 <- r1_all[rownames(edges)]

m_gr_main <- lm(log(vel) ~ axond + r1, data = edges)   # Model A
m_gr_int  <- lm(log(vel) ~ axond * r1, data = edges)   # Model B
anova(m_gr_main, m_gr_int)
m_gr_full <- lm(log(vel) ~ f + deperp + axond * r1, data = edges)
summary(m_gr_main)
summary(m_gr_int)
summary(m_gr_full)
