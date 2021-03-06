#Salva Resultados
#np.savetxt('/content/drive/MyDrive/data_CMU/Results/MSE_model_martial_30.csv', MSE_model30_t, delimiter=',')
#np.savetxt('/content/drive/MyDrive/data_CMU/Results/MSE_interp_martial_30_t.csv', MSE_interp30_t_martial, delimiter=',')

#np.savetxt('/content/drive/MyDrive/data_CMU/Results/MSE_model_martial_60.csv', MSE_model60_t_martial, delimiter=',')
#np.savetxt('/content/drive/MyDrive/data_CMU/Results/MSE_interp_martial_60_t.csv', MSE_interp60_t_martial, delimiter=',')


#Carrega Resultados
MSE_model_martial_30 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_model_martial_30.csv', delimiter=',')
MSE_interp_martial_30 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_interp30_t_martial.csv', delimiter=',')

MSE_model_martial_60 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_model_martial_60.csv', delimiter=',')
MSE_interp_martial_60 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_interp60_t_martial.csv', delimiter=',')

MSE_model_indian_30 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_model_indian_30.csv', delimiter=',')
MSE_interp_indian_30 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_interp30_t_indian.csv', delimiter=',')

MSE_model_indian_60 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_model_indian_60.csv', delimiter=',')
MSE_interp_indian_60 = np.loadtxt('/content/drive/MyDrive/data_CMU/Results/MSE_interp60_t_indian.csv', delimiter=',')

#Boxplots
fig1, ax1 = plt.subplots()
ax1.set_title('MSE Valid')
ax1.set_xlabel('Dados')

Dados = ['Model_martial30', ' Interp_martial30', 'Model_indian30', 'Interp_indian30']

ax1.set_xticklabels(Dados, rotation=45, fontsize=10)

ax1.boxplot([MSE_model_martial_30, MSE_interp_martial_30,  MSE_model_indian_30, MSE_interp_indian_30], showfliers=False)
