# Calculate cancer frequency per state
calculate.cancer.frequency.per.state = function(inputFileName) {
  stateCancer = read.csv(paste(inputFileName,".csv",sep = ""), header = F, sep = ",")
  stateCancer = stateCancer[complete.cases(stateCancer),]
  num.per.state = aggregate(stateCancer[2], by = stateCancer[1], FUN = length)
  hasCancer = stateCancer[stateCancer[,2] == 1,]
  cancer.num.per.state = aggregate(hasCancer[2],by=hasCancer[1],FUN = length)
  cancer.frequency.per.state = as.data.frame(cbind(cancer.num.per.state[,1], cancer.num.per.state[,2]/num.per.state[,2]))
  cancer.frequency.per.state.name = merge(state.code, cancer.frequency.per.state, by = "V1")
  return (cancer.frequency.per.state.name)
}

state.code = read.table("state-code.txt", sep = "\t", header = F)
file.names = c("stateCancer11","stateCancer12","stateCancer13","stateCancer14","stateCancer15")
for (file.name in file.names) {
  assign(paste("frequency.",file.name, sep = ""), calculate.cancer.frequency.per.state (file.name))
}

# Visualize in a choropleth map
state.frequency.data=frequency.stateCancer11[,2:3]
names(state.frequency.data)=c("region","value")
state.frequency.data$region=as.character(state.frequency.data$region)
state.frequency.data$value=as.numeric(state.frequency.data$value)
library(choroplethr)
state_choropleth(state.frequency.data,title = "2011",legend = "Frequency",num_colors=1)
