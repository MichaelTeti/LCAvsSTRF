library(ggplot2)
library(readr)
library(hexbin)
library(dplyr)
library(scales)
library(reshape2)
library(corrr)
library(tidyr)

#####################################################################
read_csv("thresh_grid_multiple_soft.txt") %>% 
  group_by(LCA_Timestep, Filter) %>%
  summarize(mean_lca_sparsity = mean(LCA_Sparsity)) %>%
  filter(LCA_Timestep > 0) %>%
  mutate(Filter = ifelse(Filter == "0.075", "Low Sparsity",
                         ifelse(Filter == "0.125", "Moderate Sparsity",
                                ifelse(Filter == "0.175", "High Sparsity", NA)))) %>%
  
  ggplot(lca_sp_vs_lca_ts, mapping = aes(x = LCA_Timestep, y = mean_lca_sparsity)) + 
    geom_line(aes(linetype = as.factor(Filter)), color = 'black', size = 1) + 
    theme_bw() + 
    xlab("LCA Timestep") + 
    ylab("LCA Sparsity") + 
    scale_linetype_discrete(name = "LCA Model", breaks = c("Low Sparsity", "Moderate Sparsity", "High Sparsity"))
#####################################################################
read_csv("thresh_grid_multiple_soft.txt") %>%
ggplot(mapping = aes(x = STRF_Threshold, y = LCA_Timestep)) + 
  geom_tile(mapping = aes(fill = Cossim)) + 
  xlab("LNL Threshold") + 
  ylab("LCA Timestep") + 
  scale_fill_continuous(name = "Cosine\nSimilarity", limits=(c(0, 0.6)), oob = squish, type = "viridis") + 
  theme_bw()
######################################################################
read_csv("thresh_grid_multiple_soft.txt") %>% 
  filter(LCA_Timestep == 4000)  %>%
  mutate(Model = ifelse(Filter == 0.075, "Low Sparsity", 
                        ifelse(Filter == 0.125, "Moderate Sparsity", 
                               ifelse(Filter == 0.175, "High Sparsity", NA)))) %>%
  ggplot(df, mapping = aes(x = STRF_Threshold, y = Cossim, linetype = Model)) + 
    geom_line(color = "black") + 
    theme_bw() + 
    scale_linetype(name = "Model") +
    ylab("Cosine Similarity") + 
    xlab("LNL Threshold") +
    scale_linetype_discrete(breaks = c("Low Sparsity", "Moderate Sparsity", "High Sparsity"))
####################################################################
df_ts_thresh <- 
  read_csv("thresh_grid_multiple_soft.txt") %>%
  filter(LCA_Timestep == 4000) %>%
  mutate(filter_name = ifelse(Filter == 0.075, "LS", 
                              ifelse(Filter == 0.125, "MS", 
                                     ifelse(Filter == 0.175, "HS", NA)))) 

lca_sparsity <- select(df_ts_thresh, LCA_Sparsity, filter_name, STRF_Threshold) %>%
  mutate(Model = paste("LCA-", filter_name, sep = "")) %>%
  rename(sparsity = LCA_Sparsity)

strf_sparsity <- select(df_ts_thresh, STRF_Sparsity, filter_name, STRF_Threshold) %>%
  mutate(Model = paste("LNL-", filter_name, sep = "")) %>%
  rename(sparsity = STRF_Sparsity)

df_sparsity <- rbind(lca_sparsity, strf_sparsity)

ggplot(df_sparsity, aes(x = STRF_Threshold, y = sparsity, color = Model)) +
  geom_line(size = 0.75) + 
  theme_bw() + 
  xlab("LNL Threshold") +
  ylab("Sparsity") + 
  xlim(c(0.1, 0.99)) +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) + 
  scale_color_discrete(name = "Model",
                          breaks = c("LCA-LS", "LCA-MS", "LCA-HS", "LNL-LS", "LNL-MS", "LNL-HS"))
####################################################################
read_csv("thresh_grid_multiple_ts.txt") %>%
  ggplot(aes(x = STRF_Threshold, y = as.factor(Filter))) + 
    geom_tile(mapping = aes(fill = Cossim)) + 
    theme_bw() +
    scale_fill_continuous(name = "Cosine\nSimilarity", limits = c(0.05, 0.15), oob = squish, type = "viridis") +
    xlim(0.35, 0.85) +
    ylab("LCA Threshold") + 
    xlab("LNL Threshold")
#####################################################################
###################### raw activation comparison ####################
#https://stackoverflow.com/questions/22181132/normalizing-y-axis-in-histograms-in-r-ggplot-to-proportion-by-group

# Histogram and boxplot
read_csv("LCAvsSTRF_RawActs_SoftThresh0.765.txt") %>%
  filter(grepl("HS", Key)) %>%
   filter(Act != 0.0) %>%
  ggplot(aes(x = Act, group = Key, fill = Key)) +
    stat_bin(aes(y = ..density..*0.01, group = Key), position = 'identity', alpha = 0.6, binwidth=0.01) +
    theme_bw() +
    xlab("Activation Value") + 
    ylab("Proportion Active By Model") +
    scale_fill_discrete(name = "Model") +
    theme(axis.text=element_text(size=12),
          axis.title=element_text(size=14,face="bold"))


#####################################################################
df <- read_csv("LCAvsSTRF_RawActs_SoftThresh0.765.txt") %>%
  filter(Act != 0)
df$Key <- factor(df$Key, levels = c("LCA-LS", "LCA-MS", "LCA-HS", "LNL-LS", "LNL-MS", "LNL-HS"))
ggplot(df, aes(x = Key, y = Act)) + 
  geom_boxplot(notch = TRUE, alpha = 0.1) + 
  theme_bw() +
  xlab("Model") + 
  ylab("Activation") + 
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"),
        legend.position = 'none')
#####################################################################
df <-read_csv("LCAvsSTRF_RawActs_SoftThresh0.765.txt") %>%
  filter(Key == "LCA-MS" | Key == "LNL-MS")

lca <- filter(df, Key == "LCA-MS") %>%
  rename(LCA = Act) %>%
  select(LCA)
lnl <- filter(df, Key == "LNL-MS") %>%
  rename(LNL = Act) %>%
  select(LNL)
  
df <- cbind(lca, lnl)

ggplot(df, aes(x = LNL, y = LCA)) +
  geom_jitter(alpha = 0.15) +
  theme_bw() +
  xlab("LNL Response") + 
  ylab("LCA Response") +
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold"))
