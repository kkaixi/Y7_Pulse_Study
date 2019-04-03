setwd('P:/Data Analysis/Projects/AHEC EV')
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/Lib/PMG/COM/read_data.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/Lib/PMG/COM/helper.R")
source("C:/Users/tangk/AppData/Local/Continuum/anaconda3/Lib/PMG/COM/stat_tests.R")
library(jsonlite)
library(lme4)

directory <- 'P:/Data Analysis/Projects/Y7 Pulse Study'
table <- read.csv(file.path(directory,'Table.csv'))
rownames(table) <- table$SE


model <- lmer(Chest_3ms ~ Pulse + (1|Model), data=table)
model.null <- lmer(Chest_3ms ~ (1|Model), data=table)

anova(model.null,model)


res <- aov(Chest_3ms ~ Pulse + Model, data=table)
summary(res)
