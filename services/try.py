import pandas as pd

# importing numpy as np
import numpy as np
#
# # dictionary of lists
# dict = {'First Score': [100, np.nan, np.nan, 95],
#         'Second Score': [30, np.nan, 45, 56],
#         'Third Score': [52, np.nan, 80, 98],
#         'Fourth Score': [np.nan, np.nan, np.nan, 65]}

df = pd.DataFrame({'numbers': [1, 2, 3, np.nan],
                   'colors' : ['red', 'white', 'blue', np.nan]}
                  )


#
# df = pd.DataFrame(dict)

# using dropna() function
df.dropna(how='all', inplace = True)
print(df)
print("......")