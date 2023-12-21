
library(verification)
library(plyr)
library(reshape2)
library(foreach)
library(doParallel)


train.filename   <- "data/train.csv"
test.filename    <- "data/test.csv"

dataLoad <- function(filename, nrows=-1) {
  read.table(filename, header=TRUE,
             colClasses = c("integer",rep("integer",9)),
             sep=",", nrows=nrows)
}



sigmoid <- function(x,lambda=1,shift=0) {
  x.t <- x
  1 / ( 1 + exp(-1 * lambda * (x.t+shift) ))
}


count.matches <- function(test,train,mcols) {
  rowcounts <- table(unlist(sapply(mcols,
                                   function(c) {which(train[,c]==test[c])})))
  s <- 0
  if (length(rowcounts) > 0) {
    s <- sum(rowcounts == length(mcols))
  }
  s
}


find.best.match <- function(test,train,mcols) {
  rowcounts <- table(unlist(sapply(mcols,
                                   function(c) {which(train[,c]==test[c])})))
  m <- c()
  if (length(rowcounts) > 0) {
    m <- as.numeric(names(rowcounts)[which(rowcounts==max(rowcounts))])
  }
  m
}

assoc.metrics <- function(ca.test, ca.train, base, acols) {
  
  if (nrow(ca.train) > 0) {
    # X = Matching columns
    # Y = Resource
    #
    # Want to know likelyhood that X -> Y
    #
    num.train <- count.matches(ca.test,ca.train,acols)
    num.all   <- count.matches(ca.test,base,acols)
    
    supp.X  <- num.all/nrow(base)
    supp.Y  <- nrow(ca.train)/nrow(base)
    supp.XuY <- num.train/nrow(base)
    
    confidence <- num.train/num.all
    lift <- supp.XuY/(supp.X * supp.Y)
    conviction <- (1-supp.Y)/(1-confidence)
    
  } else {
    supp.XuY   <- 0
    confidence <- 0
    lift       <- 0
    conviction <- 0
  }
  
  data.frame(Support=supp.XuY,
             Confidence=confidence,
             Lift=lift,
             Conviction=conviction)
}


comp.interest <- function(ci.test,
                          ci.train,
                          base,
                          thresh=0.5,
                          train.cols=3:10) {
  
  # Use column names.
  cols <- colnames(base)[train.cols]
  
  # First, check to see if this is a match to a
  # given observation
  exact.match <- which(apply(ci.train, 1,
                             function(r) {
                               all(r[cols]==ci.test[cols])
                             }))
  
  if (length(exact.match) > 0) {
    # Found a match in the observations. Return
    # the action.
    return(ci.train$ACTION[exact.match])
  }
  
  # Find the base rows that contain ANY of the values
  # in the test vector. Only use those rows when computing
  # the independent probability. Also make sure the
  # training set is included.
  base.rows <- unique(unlist(sapply(cols,
                                    function (i) {
                                      which(base[,i]==ci.test[[i]])
                                    })))
  
  base.rows <- base.rows[sample.int(length(base.rows),
                                    min(length(base.rows),5000))]
  base <- base[base.rows, ]
  base <- rbind(ci.train, base)
  
  ci.train.pos <- ci.train[ci.train$ACTION==1,]
  ci.train.neg <- ci.train[ci.train$ACTION==0,]
  
  # Find best matching row in positive training examples
  best.pos.cols <- c()
  best.pos <- head(find.best.match(ci.test,ci.train.pos,cols),1)
  if (length(best.pos) > 0) {
    best.pos.cols <- cols[which(ci.test[cols] == ci.train.pos[best.pos,cols])]
  }
  best.pos.cols.f <- best.pos.cols
  
  # Find best matching row in negative training examples
  best.neg.cols.f <- c()    
  if (nrow(ci.train.neg) > 0) {
    best.neg <- head(find.best.match(ci.test,ci.train.neg,cols),1)
    
    if (length(best.neg) > 0) {
      best.neg.cols <- cols[which(ci.test[cols] == ci.train.neg[best.neg,cols])]
      # Remove any common columns between positive and negative
      # training. 
      best.pos.cols.f <- setdiff(best.pos.cols,best.neg.cols)
      best.neg.cols.f <- setdiff(best.neg.cols,best.pos.cols)
    }
  }
  
  if (length(best.pos.cols.f) > 0) {
    pos.metrics <- assoc.metrics(ci.test, ci.train.pos, base, best.pos.cols.f)
    pos.conf <- pos.metrics$Confidence
    
  } else {
    pos.conf <- 0
  }
  
  if (length(best.neg.cols.f) > 0) {
    neg.metrics <- assoc.metrics(ci.test, ci.train.neg, base, best.neg.cols.f)
    neg.conf <- neg.metrics$Confidence
    
  } else {
    neg.conf <- 0
  }
  
  # Return spread between positive and
  # negative confidence.
  pos.conf - neg.conf 
}


interest.predict <- function(i.test, i.train,
                             thresh=0.7, lambda=10,
                             num.cores=1) {
  
  # Split computation into chunks for parallelization.
  resources <- unique(i.train$RESOURCE)
  resources <- resources[resources %in% unique(i.test$RESOURCE)]
  chunks <- split(1:length(resources), ((1:length(resources)) %% num.cores))
  
  preds <- ldply(1:length(chunks),
                 function(chunk.idx) {
                   ldply(chunks[[chunk.idx]], function(ridx) {
                     
                     r <- resources[ridx]
                     cii.train <- i.train[i.train$RESOURCE==r,,drop=FALSE]
                     cii.test  <- i.test[i.test$RESOURCE==r,,drop=FALSE]
                     
                     if (sum(cii.train$ACTION)==0) {
                       # There are no positive examples. For simplicity,
                       # just return 0
                       return(data.frame(id=cii.test$id, ACTION=0))
                     }
                     
                     action <- apply(cii.test, 1, function(t.test) {
                       comp.interest(t.test,
                                     cii.train,
                                     i.train,
                                     thresh)
                     })
                     
                     # Use sigmoid to fit response into range of 0 to 1
                     lambda <- 10
                     shift  <- 0
                     action <- sigmoid(action,lambda,shift)
                     
                     data.frame("id"=cii.test$id, "ACTION"=action)
                   })}, 
                 .parallel=TRUE)
  
  preds 
}

# ------------------------------------------------------------
# MAIN

main <- function() {
  
  num.cores <- 2
  registerDoParallel(cores=num.cores)
  
  train <- dataLoad(train.filename)
  test  <- dataLoad(test.filename)
  
  # Cross Validation
  #
  # main.test.cv(train)
  
  preds <- interest.predict(test,train, num.cores)
  preds <- preds[order(preds$id),]
  write.csv(preds, 
            file = format(Sys.time(), "submission_%Y.%m.%d_%H.%M.csv"),
            row.names = F, quote = F)
}


main.test.cv <- function(train) {
  
  # Use most frequent resources to build CV training set.
  r.count <- ddply(train, .(RESOURCE), nrow)  
  r.sample <- head(r.count[order(-r.count$V1),],5)
  X.sample <- train[train$RESOURCE %in% r.sample$RESOURCE,]
  
  set.seed(12356)
  test.cv(X.sample, 1)  
}

test.cv <- function(X.sample, num.cores=1) {
  
  k <- 10
  bucket.size <- floor(nrow(X.sample)/k)
  tmp.indexes <- seq(1,nrow(X.sample))
  folds       <- c()
  for (i in seq(1,(k-1))) {
    fold.indexes <- sample(tmp.indexes, bucket.size)
    tmp.indexes  <- tmp.indexes[! tmp.indexes %in% fold.indexes]    
    folds        <- c(folds, list(fold.indexes))
  }
  folds <- c(folds, list(tmp.indexes))
  
  for (i in seq(1,length(folds))) {
    X.train <- X.sample[-folds[[i]],]
    X.test  <- X.sample[ folds[[i]],]
    X.test$id <- 1:nrow(X.test)
    
    # Make the predictions
    preds <- interest.predict(X.test,X.train, num.cores)
    
    p <- preds[order(preds$id),"ACTION"]
    
    obs.df <- X.test[X.test$id %in% preds$id,]
    obs <- obs.df[order(obs.df$id),"ACTION"]
    
    pp <- pt.performance(obs,p)
    print(pp)
  }
}


pt.performance <- function(actual, probs, thresh=0.5, do.auc=TRUE) {
  
  predicted <- ifelse(probs >= thresh, 1, 0)
  
  tp <- sum(actual==1&predicted==1)
  fp <- sum(actual==0&predicted==1)
  fn <- sum(actual==1&predicted==0)
  tn <- sum(actual==0&predicted==0)
  
  tpr <- tp/(tp+fn)
  fpr <- fp/(fp+tn)
  
  errrate   <- (fp+fn)/(tp+fp+tn+fn)
  recall    <- tp/(tp+fn)
  precision <- tp/(tp+fp)
  
  tpr       <- ifelse(is.na(tpr), 0, tpr)
  fpr       <- ifelse(is.na(fpr), 0, fpr)
  errrate   <- ifelse(is.na(errrate), 0, errrate)
  recall    <- ifelse(is.na(recall), 0, recall)
  precision <- ifelse(is.na(precision), 0, precision)
  
  F1 <- 2 * (precision * recall)/(precision + recall)
  F1 <- ifelse(is.na(F1),0,F1)
  if (do.auc) {
    auc <- (roc.area(actual, probs))$A
  } else {
    auc <- 0
  }
  perf.df <- data.frame(auc,
                        tp, 
                        fp,
                        tn,
                        fn,
                        tpr,
                        fpr,
                        errrate,
                        recall,
                        F1,
                        precision)
  perf.df
}