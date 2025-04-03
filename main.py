import xml.etree.ElementTree as ET
import pandas as pd

from wwo_hist import retrieve_hist_data

# xml_file = "/Users/utkuozer/Desktop/DSA Proje/DSA-210-Project/export.xml"  
# tree = ET.parse(xml_file)
# root = tree.getroot()

# Extracting my sleep data
# sleep_data = [] 
# for record in root.findall("Record"):
#     if record.attrib.get("type") == "HKCategoryTypeIdentifierSleepAnalysis":
#         # since it is very large dataset we will search it
#         # with specific "HKCategoryTypeIdentifierSleepAnalysis" keyword.
#         sleep_data.append({
#             "start_time": record.attrib.get("startDate"),
#             "end_time": record.attrib.get("endDate"),
#             "value": record.attrib.get("value")
#         })

# Converting my sleep data to CSV
# sleep_df = pd.DataFrame(sleep_data)
# sleep_df.to_csv("sleep_data.csv", index=False)
# print("Sleep data saved to sleep_data.csv")

# will set an API KEY from openweather and will os for security.


# printing my sleep data.
df = pd.read_csv("DSA-210-Project/sleep_data.csv")
print(df.tail())

