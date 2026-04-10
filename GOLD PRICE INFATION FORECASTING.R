# ===============================
# Install & Load Required Libraries
# ===============================
install.packages(c("readxl", "forecast", "tseries", "uroot", "ggplot2", "Metrics", "dplyr"))

library(readxl)
library(forecast)
library(tseries)
library(uroot)
library(ggplot2)
library(Metrics)
library(dplyr)

search()

# Set working directory 
setwd("C:/Users/Merlin/Downloads/RESEARCH PROJECT")

# Load Excel data
df <- read_excel("Gold Price.xlsx")

# Check column names
print(names(df))

# ===============================
# Convert Date Column
# ===============================
df$Date <- as.Date(df$Date)

# SORT DATA BY DATE 
df <- df[order(df$Date), ]

# ===============================
# Monthly Conversion
# ===============================

df_monthly <- df %>%
  mutate(YearMonth = format(Date, "%Y-%m")) %>%
  group_by(YearMonth) %>%
  summarise(Log_Price = mean(Log_Price))

df_monthly$Date <- as.Date(paste0(df_monthly$YearMonth, "-01"))

# Create monthly time series
start_year  <- as.numeric(format(min(df_monthly$Date), "%Y"))
start_month <- as.numeric(format(min(df_monthly$Date), "%m"))

gold_ts <- ts(df_monthly$Log_Price,
              start = c(start_year, start_month),
              frequency = 12)

head(df_monthly$Date)
tail(df_monthly$Date)

# ===============================
# Plot Original Log Price Series
# ===============================
plot(gold_ts,
     main = "Log of Gold Price Time Series",
     xlab = "Year",
     ylab = "Log(Gold Price)",
     col = "blue")

# ===============================
# HEGY Seasonal Unit Root Test
# ===============================
hegy_result <- hegy.test(gold_ts)
print(hegy_result)

# ===============================
# ADF Test (Non-seasonal, Original Series)
# ===============================
adf_result <- adf.test(gold_ts)
print(adf_result)

# ===============================
# Train-Test Split (80%-20%)
# ===============================
n <- length(gold_ts)
train_size <- floor(0.8 * n)
train_ts <- window(gold_ts, end = time(gold_ts)[train_size])
test_ts  <- window(gold_ts, start = time(gold_ts)[train_size + 1])

# ===============================
# ARIMA Model (No Manual Differencing)
# ===============================
arima_model <- auto.arima(train_ts, seasonal = FALSE)
summary(arima_model)

arima_forecast_test <- forecast(arima_model, h = length(test_ts))

# ===============================
# Holt-Winters Model
# ===============================
hw_model <- HoltWinters(train_ts)
hw_forecast_test <- forecast(hw_model, h = length(test_ts))

# ===============================
# Model Evaluation
# ===============================
actual   <- as.numeric(test_ts)
arima_fc <- as.numeric(arima_forecast_test$mean)
holt_fc  <- as.numeric(hw_forecast_test$mean)

library(Metrics)

arima_acc <- c(
  RMSE = rmse(actual, arima_fc),
  MAE  = mae(actual, arima_fc),
  MAPE = mape(actual, arima_fc) * 100
)

hw_acc <- c(
  RMSE = rmse(actual, holt_fc),
  MAE  = mae(actual, holt_fc),
  MAPE = mape(actual, holt_fc) * 100
)

print(arima_acc)
print(hw_acc)

# ===============================
# Accuracy Comparison Table
# ===============================
accuracy_table <- rbind(
  ARIMA = arima_acc,
  Holt_Winters = hw_acc
)

colnames(accuracy_table) <- c("RMSE", "MAE", "MAPE")

print("Model Comparison:")
print(accuracy_table)

# ===============================
# Refit Models on Full Data
# ===============================
final_arima <- auto.arima(gold_ts, seasonal = FALSE)
final_hw    <- HoltWinters(gold_ts)
# ===============================
# Future Forecast (Next 12 Months)
# ===============================
arima_future <- forecast(final_arima, h = 12)
hw_future    <- forecast(final_hw, h = 12)

# ===============================
# Plot Future Forecasts
# ===============================
plot(arima_future,
     main = "Log Gold Price Forecast - ARIMA",
     xlab = "Year",
     ylab = "Log(Gold Price)")

plot(hw_future,
     main = "Log Gold Price Forecast - Holt-Winters",
     xlab = "Year",
     ylab = "Log(Gold Price)")

# ===============================
# Print Forecast Values
# ===============================
print("ARIMA Log-Price Forecast:")
print(arima_future)
print("Holt-Winters Log-Price Forecast:")
print(hw_future)

# ===============================
# Residual Extraction
# ===============================

# Residuals from final ARIMA model
arima_residuals <- residuals(final_arima)

# Residuals from final Holt model
holt_residuals <- residuals(final_hw)

# Remove NA values
arima_residuals <- na.omit(arima_residuals)
holt_residuals  <- na.omit(holt_residuals)

# Plot residuals
plot(arima_residuals,
     main = "ARIMA Residuals",
     ylab = "Residuals",
     xlab = "Time")

plot(holt_residuals,
     main = "Holt Residuals",
     ylab = "Residuals",
     xlab = "Time")

#===============================
# Residual Diagnostics
#===============================

# Histogram
hist(arima_residuals,
     breaks = 30,
     col = "lightblue",
     main = "Histogram of ARIMA Residuals",
     xlab = "Residuals")

# Q-Q Plot
qqnorm(arima_residuals)
qqline(arima_residuals, col = "red")

# Shapiro-Wilk Normality Test
shapiro.test(arima_residuals)

hist(holt_residuals,
     breaks = 30,
     col = "lightgreen",
     main = "Histogram of Holt Residuals",
     xlab = "Residuals")

qqnorm(holt_residuals)
qqline(holt_residuals, col = "red")

shapiro.test(holt_residuals)

#Autocorrelation
# ACF plot
acf(arima_residuals,
    main = "ACF of ARIMA Residuals")

# Ljung-Box Test
Box.test(arima_residuals,
         lag = 20,
         type = "Ljung-Box")

acf(holt_residuals,
    main = "ACF of Holt Residuals")

Box.test(holt_residuals,
         lag = 20,
         type = "Ljung-Box")

# ===============================
# Neural Network Modeling of Residuals
# ===============================

library(forecast)

# Neural Network on ARIMA residuals
nn_arima_resid <- nnetar(arima_residuals,
                         size = 10,
                         repeats = 20)

# Neural Network on Holt residuals
nn_holt_resid <- nnetar(holt_residuals,
                        size = 10,
                        repeats = 20)


# ===============================
# Residual Forecasting (12 Steps Ahead)
# ===============================

# Forecast residuals using Neural Networks
arima_resid_forecast <- forecast(nn_arima_resid, h = 12)$mean
holt_resid_forecast  <- forecast(nn_holt_resid, h = 12)$mean


# ===============================
# Hybrid Forecast Construction
# ===============================

# Hybrid ARIMA–NN Forecast
hybrid_arima_nn <- arima_future$mean + arima_resid_forecast

# Hybrid Holt–NN Forecast
hybrid_holt_nn <- hw_future$mean + holt_resid_forecast


# ===============================
# Visualization
# ===============================

plot(hybrid_arima_nn,
     main = "Hybrid ARIMA–Neural Network Forecast (Log Gold Price)",
     ylab = "Log(Gold Price)",
     xlab = "Time")

plot(hybrid_holt_nn,
     main = "Hybrid Holt–Neural Network Forecast (Log Gold Price)",
     ylab = "Log(Gold Price)",
     xlab = "Time")

#===============================
#  ENSEMBLED HYBRID MODEL
#===============================

# Simple average ensemble
ensemble_hybrid_forecast <- (hybrid_arima_nn + hybrid_holt_nn) / 2

# Visualization
plot(ensemble_hybrid_forecast,
     main = "Ensembled Hybrid Forecast (ARIMA–NN + Holt–NN)",
     ylab = "Log(Gold Price)",
     xlab = "Time")


# ===============================
# Convert to ORIGINAL SCALE
# ===============================

# Convert actual values (log → original)
actual <- exp(as.numeric(test_ts))

# Convert forecasts (log → original)
arima_fc_num <- exp(as.numeric(arima_forecast_test$mean))
holt_fc_num  <- exp(as.numeric(hw_forecast_test$mean))

arima_nn_num <- exp(as.numeric(hybrid_arima_nn))
holt_nn_num  <- exp(as.numeric(hybrid_holt_nn))

# Check lengths
length(actual)
length(arima_fc_num)
length(holt_fc_num)
length(arima_nn_num)
length(holt_nn_num)

# ===============================
# Accuracy Calculation (REAL SCALE)
# ===============================
accuracy_table <- rbind(
  ARIMA = c(
    RMSE = Metrics::rmse(actual, arima_fc_num),
    MAE  = Metrics::mae(actual, arima_fc_num),
    MAPE = Metrics::mape(actual, arima_fc_num) * 100
  ),
  
  Holt_Winters = c(
    RMSE = Metrics::rmse(actual, holt_fc_num),
    MAE  = Metrics::mae(actual, holt_fc_num),
    MAPE = Metrics::mape(actual, holt_fc_num) * 100
  ),
  
  ARIMA_NN = c(
    RMSE = Metrics::rmse(actual, arima_nn_num),
    MAE  = Metrics::mae(actual, arima_nn_num),
    MAPE = Metrics::mape(actual, arima_nn_num) * 100
  ),
  
  Holt_NN = c(
    RMSE = Metrics::rmse(actual, holt_nn_num),
    MAE  = Metrics::mae(actual, holt_nn_num),
    MAPE = Metrics::mape(actual, holt_nn_num) * 100
  )
)

print("Accuracy Table (Original Scale):")
print(accuracy_table)

# ===============================
# Best Model Selection
# ===============================
best_model <- rownames(accuracy_table)[which.min(accuracy_table[, "RMSE"])]
cat("Best Model Based on RMSE:", best_model)

# ===============================
# Final Forecast (ORIGINAL SCALE)
# ===============================
final_forecast <- switch(as.character(best_model),
                         ARIMA        = exp(arima_future$mean),
                         Holt_Winters = exp(hw_future$mean),
                         ARIMA_NN     = exp(hybrid_arima_nn),
                         Holt_NN      = exp(hybrid_holt_nn),
                         stop("Unknown best_model value")
)

print("Final Forecast (Original Gold Price):")
print(final_forecast)

# ===============================
# Plot (ORIGINAL SCALE)
# ===============================
if (best_model == "ARIMA") {
  plot(exp(arima_future$mean),
       main = "Final Forecast using ARIMA (Original Scale)",
       ylab = "Gold Price",
       xlab = "Time")
  
} else if (best_model == "Holt_Winters") {
  plot(exp(hw_future$mean),
       main = "Final Forecast using Holt-Winters (Original Scale)",
       ylab = "Gold Price",
       xlab = "Time")
  
} else if (best_model == "ARIMA_NN") {
  plot(exp(as.ts(hybrid_arima_nn)),
       main = "Final Forecast using ARIMA-NN (Original Scale)",
       ylab = "Gold Price",
       xlab = "Time")
  
} else if (best_model == "Holt_NN") {
  plot(exp(as.ts(hybrid_holt_nn)),
       main = "Final Forecast using Holt-NN (Original Scale)",
       ylab = "Gold Price",
       xlab = "Time")
}

print(paste("Final Forecast using", best_model))
print(final_forecast)
