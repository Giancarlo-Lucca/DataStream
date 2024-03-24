# install.packages("stream")
library("stream")
library("streamMOA")

execute_exp <- function(alg, name_ex){
  path = "/home/asier.urio/Ikerketa/Projects/DataStream/OtherAlgorithms"
  dataset = "Benchmark1_11000R.csv"
  iterations <- 11
  chunksize <-1000
  nClases <- 2
  dataset = "SamplesFile_b_4C2D800Linear.csv"
  iterations <- 8
  chunksize <-100
  nClases <- 4
  file <- paste(path,dataset,sep="/")
  out_p <- paste(path,gsub(".csv","",dataset),sep="/")
  out <- paste(out_p,name_ex,".csv",sep="")
  
  train_stream <- DSD_ReadCSV(file,k=nClases,header=TRUE,sep=",")
  eval_stream <- DSD_ReadCSV(file,k=nClases,header=TRUE,sep=",")
  reset_stream(train_stream)
  reset_stream(eval_stream)
  
  stream_alg <- alg
  
  Data <- matrix(ncol=1, nrow=iterations)
  # print(paste("Evaluation","ARI","Silhouette"))
  for (x in 1:iterations){
    update(stream_alg, train_stream, n = chunksize)
    es <- evaluate_static(stream_alg, eval_stream, measure=c("cRand"), n = chunksize)
    row <- list(as.numeric(es[1]))# ,as.numeric(es[2]))
    Data[x,1] <- as.numeric(es[1])
    # Data[x,2] <- as.numeric(es[2])
    # print(paste(x," evaluation",Data[x,1],Data[x,2]))
  }
  
  close_stream(train_stream)
  close_stream(eval_stream)
  
  Data <- data.frame(Data)
  names(Data)[1] <- "cRand"
  # names(Data)[2] <- "silhouette"
  write.csv(Data,out)
  # print(mean(Data[,"cRand"]))
  
  return(mean(Data[,"cRand"]))
}

getArgs <- function(){
  args <- commandArgs(trailingOnly = TRUE)
  dataset <- args[2]
}


main <- function(){
  
  # print("DBStream")
  best <- 0
  best_a <- -1
  best_l <- -1
  for (a in c(0.001, 0.003, 0.01, 0.1, 0.5, 0.7, 0.9, 1)){
    for (l in c(0.001, 0.003, 0.01, 0.1, 0.5, 0.7, 0.9, 1)){
        dbstream <- DSC_DBSTREAM(
          "formula"=NULL,
          "r"=2,
          "lambda"=as.numeric(l),
          "gaptime"=1000L,
          "Cm"=3,
          "metric"="Euclidean",
          "noise_multiplier"=1,
          "shared_density"=FALSE,
          "alpha"=as.numeric(a),
          "k"=0,
          "minweight"=0
        )
        dbstream_out <- execute_exp(dbstream, "dbstream") 
        if (dbstream_out > best){
          best <- dbstream_out
          best_a <- a
          best_l <- l
        }
    }
  }

  print(paste("DBSTream Best:",best, best_a, best_l))
  # dbstream_out <- execute_exp(dbstream, "dbstream") 
  
# print("DStream")
  best <- 0
  best_g <- -1
  best_l <- -1
  gsizes <- c(0.1, 0.2, 0.3) #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) 
  for (g in gsizes){
    for (l in c(0.001, 0.003, 0.01, 0.1, 0.5, 0.7, 0.9, 1)){
    dstream <- DSC_DStream(
      formula=NULL,
      gridsize=as.numeric(g),
      lambda=as.numeric(l), #0.003 2^(-lmabda) = 0.998
      gaptime=100L,
      Cm=1,
      Cl=0.8
    )#beta 0.3
    
    dstream_out <- execute_exp(dstream, "dstream") 
    if (dstream_out > best){
      best <- dstream_out
      best_g <- g
      best_l <- l
    }
    }
  }
  print(paste("DStream Best:",best, best_g, best_l))
  
  hierac <- DSC_Hierarchical(
    formula=NULL,
    k= 2,
    h=NULL,
    method="complete",
    min_weight = NULL,
    description=NULL
  )
  print("Hierarchical")
  hierac_out <- execute_exp(hierac, "hierarc") 
  
  clustream <- DSC_CluStream(
    m = 50,
    horizon = 1000,
    t = 1,
    k = 2
  ) 
  
  print("CluStream")
  clustream_out <- execute_exp(clustream, "clustream")
  
  # print("DenStream")
  best <- 0
  best_e <- -1
  best_b <- -1
  for (eps in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)){
    for (beta in c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)){
      denstream <- DSC_DenStream(
        epsilon=eps,
        mu = 1,
        beta=beta,
        lambda=0.001,
        initPoints = 100,
        offline=2,
        processingSpeed = 1,
        recluster=TRUE,
        k = 4
      ) 
      denstream_out <- execute_exp(denstream, "denstream") 
      if (denstream_out > best){
        best <- denstream_out
        best_e <- eps
        best_b <- beta
      }
    }
  }
  print(paste("DenStream Best:",best, best_e, best_b))
}


main()