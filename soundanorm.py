#################################################

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
# %matplotlib inline

#from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# =====================================================================
# 音声データをpython speech featuresで配列変換
(rate,sig) = wav.read("normL.wav")

print("sig:",sig,",len:",len(sig),",type:",type(sig))
print("rate:",rate)

fbank_feat = logfbank(sig,rate,winlen=0.01,nfilt=40)


# =====================================================================
# 学習データとテストデータは60000データくらいのうちの前半30000でトレーニング、後半30000でトレーニングとしておく。適当。
n_traintest_sep = 300

X_trn = fbank_feat[:n_traintest_sep][:]
X_tst = fbank_feat[n_traintest_sep:][:]

# 値の範囲を[0,1]に変換
scaler = MinMaxScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# 入力データの次元数(=40)を取得
n_dim = X_trn.shape[1]


# =====================================================================
# 学習履歴をプロットする関数

# 損失関数値の履歴のプロット
def plot_history_loss(rec):
    plt.plot(rec.history['loss'],"o-",label="train",)
    plt.plot(rec.history['val_loss'],"o-",label="test")
    plt.title('loss history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()



# =====================================================================
# AutoEncoderの構築

ae = Sequential()
ae.add(Dense(36, input_dim = n_dim, activation='relu'))
ae.add(Dense(18, activation='relu'))
ae.add(Dense(9, activation='relu', name = 'encoder'))
ae.add(Dense(18, activation='relu'))
ae.add(Dense(36, activation='relu'))
ae.add(Dense(n_dim, activation='sigmoid'))

ae.compile(loss = 'mse', optimizer ='adam')
records_ae = ae.fit(X_trn, X_trn,
                    epochs=100,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_tst, X_tst))

# 学習済み重みの保存
#ae.save_weights('autoencoder.h5')
# ネットワークの概要
ae.summary()
# 損失関数値の履歴のプロット
plot_history_loss(records_ae)

#testデータを入力してみて、MSEを見てみる
output_tst = ae.predict(X_tst)
mse_tst = mean_squared_error(X_tst, output_tst )
print("TestData mse:",mse_tst)

# 別の正常音データを入力してみて、MSEを見てみる
(rate_reg,sig_reg) = wav.read("norm1.wav")
fbank_feat_reg = logfbank(sig_reg,rate_reg,winstep=0.01,nfilt=40)
X_reg = scaler.transform(fbank_feat_reg)

output_reg = ae.predict(X_reg)
mse_reg = mean_squared_error(X_reg, output_reg )
print("RegularityData1 mse:",mse_reg)

# 別の正常音データを入力してみて、MSEを見てみる2
(rate_reg,sig_reg) = wav.read("norm2.wav")
fbank_feat_reg = logfbank(sig_reg,rate_reg,winstep=0.01,nfilt=40)
X_reg = scaler.transform(fbank_feat_reg)

output_reg = ae.predict(X_reg)
mse_reg = mean_squared_error(X_reg, output_reg )
print("RegularityData2 mse:",mse_reg)

# 異常音データを入力してみて、MSEを見てみる
(rate_ano,sig_ano) = wav.read("anor1.wav")
fbank_feat_ano = logfbank(sig_ano,rate_ano,winstep=0.01,nfilt=40)
X_ano = scaler.transform(fbank_feat_ano)

output_ano = ae.predict(X_ano)
mse_ano = mean_squared_error(X_ano, output_ano)
print("AnomalyData1 mse:",mse_ano)

# 異常音データを入力してみて、MSEを見てみる2
(rate_ano,sig_ano) = wav.read("anor2.wav")
fbank_feat_ano = logfbank(sig_ano,rate_ano,winstep=0.01,nfilt=40)
X_ano = scaler.transform(fbank_feat_ano)

output_ano = ae.predict(X_ano)
mse_ano = mean_squared_error(X_ano, output_ano)
print("AnomalyData2 mse:",mse_ano)

# 異常音データを入力してみて、MSEを見てみる3
(rate_ano,sig_ano) = wav.read("anor3.wav")
fbank_feat_ano = logfbank(sig_ano,rate_ano,winstep=0.01,nfilt=40)
X_ano = scaler.transform(fbank_feat_ano)

output_ano = ae.predict(X_ano)
mse_ano = mean_squared_error(X_ano, output_ano)
print("AnomalyData3 mse:",mse_ano)