import numpy as np

weather = np.load("weather.npy")

#print(np.shape(all_data[0][]))

main_dict = {}
main_dict['Clear'] = 0
main_dict['Clouds'] = 1
main_dict['Rain'] = 2

predict_input_len = 40
predict_output_len = 8

main_input_list = []
main_output_list = []

for i in range(len(weather)):
	for j in range(len(weather[i])-predict_output_len-predict_input_len):
		input_list = []
		for k in range(predict_input_len):
			temp_list = [weather[i][j+k][3]/50,weather[i][j+k][6]/1000,weather[i][j+k][7]/50,main_dict[weather[i][j+k][9]]/2,weather[i][j+k][13]]
			input_list.append(temp_list)
		main_input_list.append(input_list)

for i in range(len(weather)):
	for j in range(predict_output_len,len(weather[i])-predict_input_len):
		output_list = []
		for k in range(predict_output_len):
			temp_list = [weather[i][j+k][13]]
			output_list.append(temp_list)
		main_output_list.append(output_list)

main_input_array = np.array(main_input_list)
main_output_array = np.array(main_output_list)

print(np.shape(main_input_array))
print(np.shape(main_output_array))

#np.save('input_weather.npy',main_input_array)
np.save('output_rain.npy',main_output_array)