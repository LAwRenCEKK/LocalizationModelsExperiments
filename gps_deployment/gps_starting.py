
# def gps_fromxy(lat0,lon0,x_offset,y_offset):
#     """ Convert back to the GPS cooridnate system from the local system.
#     Args:
#         lat0,lon0: latitude and longitude of the reference point(origin)
#         x_offset,y_offset: The offset in X-axis and Y-axis
#     Returns:
#         GPS cooridnate(longitude and latitude)
#     """
#     R = 6378137.0
#     #offset in radians
#     dlat = y_offset/R
#     dlon = x_offset/(R * math.cos(math.pi * lat0 /180.0))
#     #offset gps in decimal degrees
#     lat1 = lat0 + dlat * 180/math.pi
#     lon1 = lon0 + dlon * 180/math.pi
#     return [lon1,lat1]


def gps_start(training_data_file,num_of_unlabeled):
    import json
    import math
    import numpy as np
    import pandas as pd
    from gps_deployment.main_controller import MainController

    print (str(training_data_file))
    # Main function 
    mc = MainController(num_ap=5, training_file = training_data_file)
    mc.load_data()
    mc.initialize_gaussian_processes(visualize=False)

    # Visualizing the Gaussian processes
    predicted_rssi = [] # predictive results from each GP
    for ap_index in range(0,mc.number_of_access_points):
        predicted_rssi.append(mc.visualize_gaussian(ap_index, 
            mc.gaussian_processes[ap_index]))
    predicted_rssi = np.array(predicted_rssi)
    constant = 0.01
    # print (predicted_rssi)
    print (predicted_rssi.shape)

    
    wifi_locations = pd.read_csv(training_data_file,header=None).iloc[:,-2:]
    Y_train_rounded = np.round(wifi_locations)
    # Y_train_rounded = wifi_locations.to_numpy()

    Y_train_rounded = Y_train_rounded.to_numpy()
    final = Y_train_rounded
    temp = np.zeros((len(Y_train_rounded),2))
    for i in range(-3,2):
        for j in range(-3,2):
            if not ((i==0) and (j==0)):
                temp[:,0] = Y_train_rounded[:,0] + i*constant
                temp[:,1] = Y_train_rounded[:,1] + j*constant
                final = np.append(final,temp,axis=0)
    # final_list = final.tolist()
    final_uniques = np.unique(final,axis=0)
    locations = final_uniques
    # print (final_uniques)
    # # lat, long candidates
    # X_range = np.arange(0,100,5)
    # Y_range = np.arange(0,100,5)
    # # X_range = np.arange(-7700,-7580,5)
    # # Y_range = np.arange(4864890,4865010,5)
    # locations = []
    # for i in range(0,len(X_range)):
    #     for j in range(0,len(Y_range)):
    #         x = X_range[i]
    #         y = Y_range[j]
    #         locations.append([x,y])
    # locations = np.array(locations)
    # print (locations)

    # # Convert the location infomation from meter to lat,long
    # locations_lat_long = []
    # gps_origin = (-79.9214068, 43.2584115) # this is only ITB
    # for i in range(10000):
    #     locations_lat_long.append(
    #         gps_fromxy(
    #             gps_origin[1],
    #             gps_origin[0],
    #             locations[i][1],
    #             locations[i][0])
    #         )

    # joint converted location and predicted rssi to form fingerprint
    fingerprint = {}
    fingerprint['result']=[]
    for i in range(len(locations)):
        fingerprint['result'].append(
            {'loc': locations[i].tolist(),
            'rssi': predicted_rssi[:,i].tolist()}
            )
    # output the fingperprint map to a json file
    with open('fingerprint_map_10_{}.txt'.format(num_of_unlabeled), 'w') as outfile:
        json.dump(fingerprint, outfile)
    print (len(fingerprint['result']))
    return 0


# if __name__ == '__main__':
#     main()

