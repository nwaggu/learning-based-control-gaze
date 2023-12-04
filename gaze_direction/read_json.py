import json
import numpy as np
import glob
import os

# openpose JSON files are stored here
folder_path = "C:\\Users\\Ali\\Documents\\openpose\\output_json_folder\\"

def get_head_locs_from_json():
  # grab the most recent file
  list_of_files = glob.glob(folder_path+'*') 
  latest_file = max(list_of_files, key=os.path.getctime)
  # pull the contents as a dictionary
  with open(latest_file) as user_file:
    file_contents = json.load(user_file)
  # the split up body part locations are stored under the "part_candidates" key,
    # grab the contents of that list
  part_candidates = file_contents["part_candidates"][0]
  # the (x,y,confidence) of each detected head position is stored the "0" key
  head_locations = part_candidates["0"]
  head_locations = np.reshape(head_locations, (-1,3))

  xlocs = []
  ylocs = []
  confidence_threshold = 0  # set this based on testing?
  for x,y,confidence in head_locations:
    if confidence > confidence_threshold:
      xlocs.append(x)
      ylocs.append(y)

  return xlocs, ylocs

if __name__ == "__main__":
  from sim2D import SIM2D

  xlocs, ylocs = get_head_locs_from_json()
  # print(np.column_stack((xlocs, ylocs)))

  gridworld = SIM2D(xlocs, ylocs)
  gridworld.make_plot()

  
