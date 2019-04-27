
#loop through files and create fst equivalents
library(fst)
library(data.table)
library(pbapply)
library(dplyr)

dirs = list(
  trips = "E:/old_taxi_data/trip_data",
  fares = "E:/old_taxi_data/trip_fare",
  trip_fst = "E:/old_taxi_data/trip_data_fst",
  fares_fst = "E:/old_taxi_data/trip_fare_fst"
)


setwd(dirs$trips)
1:length(list.files()) %>%
  pblapply(function(x){
    
    #trips rewrite
    setwd(dirs$trips)
    z = list.files()[x]
    q = fread(z)
    setwd(dirs$trip_fst)
    write.fst(q, paste0("trip_",x,".fst"), compress = T)
    rm(q)
    gc()
    
    #fare rewrite
    setwd(dirs$fares)
    z = list.files()[x]
    q = fread(z)
    setwd(dirs$fares_fst)
    write.fst(q, paste0("trip_fare_",x,".fst"), compress = T)
    rm(q)
    gc()
    
  })
  
