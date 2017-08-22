#### kaggle데이터 K-NN #########

setwd("E:\\koo\\Rdata\\paysim1")
synFin <- read.csv("PS_20174392719_1491204439457_log.csv",stringsAsFactors=F)


# nameDest 'M'으로 시작하는 건 의미없음
findata <- synFin[grep("^C", synFin$nameDest),]
findata <- findata[,-1]

# install.packages("dplyr")
library(dplyr)
# filter(findata, isFraud == 1 & isFlaggedFraud == 0)

# 그룹별 라벨링 컬럼추가
lbdata0 <- mutate(filter(findata, isFlaggedFraud == 0 & isFraud == 0), label = "Good")
lbdata1 <- mutate(filter(findata, isFlaggedFraud == 0 & isFraud == 1), label = "Fraud")
lbdata2 <- mutate(filter(findata, isFlaggedFraud == 1 & isFraud == 0), label = "Flagged")
lbdata3 <- mutate(filter(findata, isFlaggedFraud == 1 & isFraud == 1), label = "FlaggedFraud")

lbdata <- rbind(lbdata0, lbdata1, lbdata2, lbdata3)
rm(lbdata0, lbdata1, lbdata2, lbdata3)

# type을 숫자로 변환
# CASH_IN: 0
# CASH_OUT: 1
# DEBIT: 2
# PAYMENT: 3
# TRANSFER: 4


# factor로 변환
#lbdata$nameOrig <- as.factor(lbdata$nameOrig)
#lbdata$nameDest <- as.factor(lbdata$nameDest)
lbdata <- lbdata[,-3] # 고객명 삭제
lbdata <- lbdata[,-5]

lbdata$type <- as.factor(lbdata$type)
lbdata$label <- as.factor(lbdata$label)

lbdata$type <- unclass(lbdata$type) # factor변수를 integer vector로 만든다.
lbdata$label <- unclass(lbdata$label) # factor변수를 integer vector로 만든다.


cdata <- lbdata[,-7:-8] # isFF/ FF 삭제


##### 데이터 건수 축소를 위한 랜덤 샘플링 

flag <- sample(c("f1", "f2"), nrow(cdata), replace = TRUE, prob = c(0.3, 0.7))
newSet1 <- cdata[flag == "f1", ] 
newSet2 <- cdata[flag == "f2", ]

rm(newSet2)
rm(flag)
##########################################################
# 재현성을 위한 seed 설정 
set.seed(123)
# idx 설정 
idx <- sample(x = c("train", "valid", "test"), 
size = nrow(newSet1), replace = TRUE, 
prob = c(3, 1, 1)) 

# idx에 따라 데이터 나누기 
train <- newSet1[idx == "train", ] 
valid <- newSet1[idx == "valid", ] 
test <- newSet1[idx == "test", ]

# install.packages("scales")
library(scales)


######################

# x와 y로 나누기
# 
# train 을 train_x, train_y로,
# valid를 valid_x, valid_y로, 
# test를 test_x, test_y로 나누자

# x는 9번째 열(label)을 제외한다는 의미로 -7
train_x <- train[, -7] 
valid_x <- valid[, -7] 
test_x <- test[, -7] 

# y는 9번째 열(label)만 필터링한다는 의미로 7
train_y <- train[, 7] 
valid_y <- valid[, 7] 
test_y <- test[, 7]


# knn 함수를 사용하기 위해 class 패키지를 설치하고 라이브러리 불러오기
# install.packages("class")
library(class)


########################

# 최적의 K를 찾는다
# 분류 정확도 사전 할당 
accuracy_k <- NULL 
# kk가 1부터 train 행 수까지 증가할 때 (반복문) 
for(kk in c(1:100)) { 
      # k가 kk일 때 knn 적용하기 
      set.seed(1234) 
      knn_k <- knn(train = train_x, 
      test = valid_x, 
      cl = train_y, 
      k = kk) 

# 분류 정확도 계산하기 
accuracy_k <- c(accuracy_k, sum(knn_k == valid_y) / length(valid_y)) 
} 

# k에 따른 분류 정확도 데이터 생성 
valid_k <- data.frame(k = c(1:100), accuracy = accuracy_k) 

# k에 따른 분류 정확도 그래프 그리기 
plot(formula = accuracy ~ k, data = valid_k, type = "o", pch = 20, main = "validation - optimal k") 

# 그래프에 k 라벨링 하기 
with(valid_k, text(accuracy ~ k, labels = rownames(valid_k), pos = 1, cex = 0.7)) 

# 분류 정확도가 가장 높으면서 가장 작은 k는? 
min(valid_k[valid_k$accuracy %in% max(accuracy_k), "k"])





########################



# R 수행시간 측정
# check excution time #1
start.time <- Sys.time()
########################

# 수행로직
# 3-NN
set.seed(1234)
knn_3 <- knn(train = train_x, 
test = valid_x, 
cl = train_y, 
k = 3) 

# check excution time #2
########################
end.time <- Sys.time()
time.taken <- end.time - start.time
print(time.taken)



# 분류 정확도 계산하기 
accuracy_3 <- sum(knn_3 == valid_y) / length(valid_y) ; 
accuracy_3

####################################
# 3-NN에 test 데이터 적용하기
set.seed(1234) 


knn_3_test <- knn(train = train_x, 
test = test_x, 
cl = train_y, 
k = 3) 

# Confusion Matrix 틀 만들기 
result <- matrix(NA, nrow = 3, ncol = 3) # label이 3개이므로 

##
# rownames(result) <- paste0("real_", levels(train_y)) 
# colnames(result) <- paste0("clsf_", levels(train_y)) 

# Confusion Matrix 값 입력하기 
result[1, 1] <- sum(ifelse(test_y == 1 & knn_3_test == "1", 1, 0)) 
result[2, 1] <- sum(ifelse(test_y == 2 & knn_3_test == "1", 1, 0)) 
result[3, 1] <- sum(ifelse(test_y == 3 & knn_3_test == "1", 1, 0))

result[1, 2] <- sum(ifelse(test_y == 1 & knn_3_test == "2", 1, 0)) 
result[2, 2] <- sum(ifelse(test_y == 2 & knn_3_test == "2", 1, 0)) 
result[3, 2] <- sum(ifelse(test_y == 3 & knn_3_test == "2", 1, 0))

result[1, 3] <- sum(ifelse(test_y == 1 & knn_3_test == "3", 1, 0)) 
result[2, 3] <- sum(ifelse(test_y == 2 & knn_3_test == "3", 1, 0)) 
result[3, 3] <- sum(ifelse(test_y == 3 & knn_3_test == "3", 1, 0))


# Confusion Matrix 출력하기 
result 

# 최종 정확도 계산하기 
sum(knn_3_test == test_y) / sum(result)
