
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({"Price":[1,2,3,4,5,6,3,5], "Car Model":[1,2,3,4,5,6,7,8]})

# # Sort the DataFrame in ascending order based on the 'Price' column
# df = df.sort_values(by='Price', ascending=True)

# # Extract the first 5 rows from the sorted DataFrame
# df_top5 = df.head(5)

# # Create a pie plot with the 'Price' column as the data and the 'Car Model' column as the labels
# plt.pie(df_top5['Price'], labels=df_top5['Car Model'])

# # Set the legend to show the car models
# plt.legend(loc='upper left')

# # Save the pie plot to 'car_specs_Num7.png'
# plt.savefig('car_specs_Num7.png')

# # Print the final result
# print("The top 5 car models by price are:")
# print(df_top5['Car Model'])


# df.sort_values('Price', ascending=True).head(5).plot.pie(y='Car Model', figsize=(10, 10), autopct='%1.1f%%', legend=True)
# plt.savefig('car_specs_Num7.png')


# print(df.corr()['Price']['Car Model'] - 0.3)
