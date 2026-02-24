
import pandas as pd
import numpy as np


def latlon_to_xyz( lat, lon , R = 1.0 ):
    """
    Given a latitude and longitude (and eventually a sphere radius),
    return the corresponding 3D coordinates (z is the axis that goes from the south to the north pole).

    NB: this function can work in a vectorized fashion

    :param lat: latitude (float or np.array)
    :param lon: longitude (float or np.array)
    :param R = 1.0: sphere radius (float)
    :return: list of x,y,z coordinates
    """
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return [x,y,z]

def xyz_to_latlon( x , y , z , R = 1.0 ):
    """
    Given a set of 3D coordinates and eventually a sphere radius),
    return the corresponding latitude and longitude.

    NB1: the z-axis is the axis that goes from the south to the north pole.
    NB2: this function can work in a vectorized fashion

    :param lat: latitude (float or np.array)
    :param lon: longitude (float or np.array)
    :param R = 1.0: sphere radius (float)
    :return: list of x,y,z coordinates
    """
    lat = np.arcsin(z / R)
    lon = np.arctan2(y, x)
    return [lat,lon]


def bearing_at_p1(p1, p2):
    """ This function computes the bearing (i.e. course) at p1 given a destination of p2.  

    :param p1: tuple point of (lon, lat)
    :param p2: tuple point of (lon, lat)
    :return: Course, in degrees
    """
    lon1, lat1 = p1
    lon2, lat2 = p2.T
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    course = np.arctan2(y, x)
    return course

def bearing_at_p1_bis(p1, p2 ,clat2 , slat2 ):
    """ This function computes the bearing (i.e. course) at p1 given a destination of p2.  

    This function uses a precomputed quantities clat2 , slat2 in order to speed up repeathed computation of bearings 

    :param p1: tuple point of (lon, lat)
    :param p2: tuple point of (lon, lat)
    :return: Course, in degrees
    """
    lon1, lat1 = p1
    lon2, lat2 = p2.T
    x = np.cos(lat1) * slat2 - np.sin(lat1) * clat2 * np.cos(lon2 - lon1)
    y = np.sin(lon2 - lon1) * clat2
    course = np.arctan2(y, x)
    return course


def point_given_start_and_bearing(p1, course , distance ):
    """ Given a start point, initial bearing, and distance, this will calculate the destination point and final
    bearing travelling along a (shortest distance) great circle arc.
    :param p1: tuple point of (lon, lat)
    :param course: Course, in degrees
    :param distance: a length in unit
    :return: point (lon, lat)
    """
    lon1, lat1 = p1
    brng = course
    delta = distance 
    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(brng))
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(delta) * np.cos(lat1), np.cos(delta) - np.sin(lat1) * np.sin(lat2))
    lon2 = (lon2 + 3*np.pi) % (np.pi*2) - np.pi
    return lon2,lat2



def get_dist_from_anchor( anchor_angles , weights ):
    '''Given a set of anchor angles and weights for each of these angles,
    this function computes a quantity between 0 and 1 representing how much (ie, by which distance) 
    is the point pulled by these different anchors.

    :param anchor_angles: angles of the anchors, in radians (numpy.array)
    :param weights: weight of each anchor (numpy.array)
    :return: "distance moved" between 0 and 1 (float)
    '''
    centroid = np.array([(np.cos( anchor_angles ) * weights).sum() , (np.sin(anchor_angles) * weights).sum()])
    return (centroid**2).sum()**0.5


def circular_mean(weights, angles):
    """this function compute a weighted average of angles
    :param weight: numpy array containing weights for each angle
    :param angles: numpy array containing angles (in radians)
    :return: weighted average angle (float)  
    """
    x = np.sum( np.cos(angles) * weights )
    y = np.sum( np.sin(angles) * weights )   
    return np.arctan2(y, x)


def spherical_relax_round(points_ll , nb_sectors , learning_rate , point_weights=None ):
    """
    :param points_ll: array containing points longitude and latitude (numpy.array)
    :nb_sectors: number of sectors to consider around each point (int)
    :learning_rate: Controls the rate at which positions are updated 
    :param point_weights: optional array of per-point repulsion weights (numpy.array).
        When provided, each neighbor's contribution to a sector is its weight rather
        than 1, so higher-weight points push surrounding points away more strongly.
        If None, all points are treated equally.

    :return: updated point longitude and latitude after a round of relaxation (numpy.array)
    """
    nb_points = len( points_ll )
    stp = np.pi*2 / nb_sectors
    anchor_angles = np.arange( 0.5 * stp, np.pi*2  , stp ) 

    if point_weights is None:
        point_weights = np.ones( nb_points )

    total_weight = point_weights.sum()

    clat = np.cos( points_ll[:,1] ) 
    slat = np.sin( points_ll[:,1] )


    new_points = []
    for i in range(nb_points):

        current_pt = points_ll[i]
        
        angles = bearing_at_p1_bis(current_pt, points_ll , clat , slat ) 

        
        sectors = ((np.array( angles+np.pi ) / (2*np.pi))*nb_sectors).astype(int)
        sectors = np.clip( sectors, 0, nb_sectors - 1 )  # safety clamp for floating-point edge cases

        # Sum each neighbor's weight into its sector.  Exclude the self-contribution
        # so a point's own weight does not bias its own movement direction.
        sector_weight_sum = np.bincount( sectors, weights=point_weights, minlength=nb_sectors )
        sector_weight_sum[ sectors[i] ] -= point_weights[i]

        denom = total_weight - point_weights[i]
        weights = sector_weight_sum / denom if denom > 0 else sector_weight_sum
        if weights.sum() > 0:
            weights /= weights.sum()

        dist = get_dist_from_anchor( anchor_angles , weights ) ## 
        bearing = circular_mean(weights, anchor_angles)

        new_pt = point_given_start_and_bearing( current_pt ,
                                               course = bearing ,
                                               distance= dist * learning_rate * np.pi/2  )
        new_points.append( new_pt )
    new_points = np.array( new_points )
    return new_points





if __name__ == "__main__":


    import sys
    import argparse

    parser = argparse.ArgumentParser(
                description="""relaxes points plotted on a sphere""")
    parser.add_argument('-i','--input-file', type=str, required=True,
             help='''input points file in csv format.
              * Expects 3 columns that should be named x, y, and z, and an index column.
              * The points are expected to be on the surface of a sphere centered on (0,0,0).
              ''')
    parser.add_argument('-o','--output-prefix', type=str, required=True,
             help='''output file prefix for the relaxed points. 
             The output file names will take the shape <output-prefix>_round<roundNumber>.csv''')

    parser.add_argument('-n','--number-rounds', type=int, default=20,
             help='number of relaxation rounds to play (default=20)')

    parser.add_argument('-w','--write-every', type=int, default=1,
             help='frequency at which the relaxed points should be written to a file; 0 means that only the result of the last round will be written (default=1, meaning that it writes to a file every round).')

    parser.add_argument('-s','--number-sectors', type=int, default=8,
             help='number of sectors to consider around the points (default=8). Higher values increase point movements precision at a computational cost.')
    parser.add_argument('-l','--learning-rate', type=float, default=0.1,
             help='''Controls the rate at which positions are updated from round to round. Lower values increase point movements precision but more relaxation rounds may be needed.
             Values should be in the interval (0,1].''')

    parser.add_argument('--weight-column', type=str, default=None,
             help='''Name of a column in the input CSV to use as per-point repulsion weights.
             Points with higher weights repel their neighbors more strongly, acquiring more
             space on the sphere surface.  If not specified all points are treated equally.
             Weights are internally normalised so that their mean equals 1.''')


    args = parser.parse_args()


    nb_sectors = args.number_sectors
    if nb_sectors < 2:
        print(f"Please specify a number of sectors >1 (option -s).")
        exit(1)

    learning_rate = args.learning_rate
    if learning_rate <= 0 or learning_rate > 1 :
        print(f"Please specify a learning rate in the interval (0,1] (option -l).")
        exit(1)

    nb_rounds = args.number_rounds
    if nb_rounds <= 0 :
        print(f"Please specify number of rounds > 0 (option -n).")
        exit(1)

    write_every = args.write_every
    if write_every < 0:
        print(f"Please specify writing frequency >= 0 (option -w).")
        exit(1)
    if write_every == 0: ## write only the last round
        write_every = nb_rounds

    
    point_file = args.input_file


    ## XYZ points 
    df_pts = pd.read_csv( point_file,
                         index_col = 0)

    for c in 'xyz':
        if not c in df_pts:
            print(f"Column {c} not found.")
            print("The input csv file should have an index and 3 columns named x,y and z.")
            exit(1)

    ## keep track of the additonal columns to include them in the output files
    additional_columns = []
    for c in df_pts.columns:
        if not c in 'xyz':
            additional_columns.append(c)

    ## load per-point repulsion weights if requested
    point_weights = None
    if args.weight_column is not None:
        if args.weight_column not in df_pts.columns:
            print(f"Weight column '{args.weight_column}' not found in input file.")
            print(f"Available columns: {list(df_pts.columns)}")
            exit(1)
        point_weights = df_pts[args.weight_column].values.astype(float)
        if point_weights.min() < 0:
            print("Warning: negative weights detected; they will be clipped to 0.")
            point_weights = np.clip(point_weights, 0, None)
        mean_w = point_weights.mean()
        if mean_w == 0:
            print("Error: all weights are zero.")
            exit(1)
        point_weights = point_weights / mean_w  # normalise so mean weight == 1
        print(f"Using weight column '{args.weight_column}' (normalised to mean=1).")





    ## sphere radius - presuming that the sphere is centered on 0
    R = np.mean( np.sqrt( df_pts.x**2 + df_pts.y**2 + df_pts.z**2 ) )

    ## XYZ coordinates to latitude, longitude
    tmp = df_pts[ ['x','y','z'] ].apply( lambda row : xyz_to_latlon( row.x , row.y , row.z , R ) , axis=1 ) 
    df_latlon = pd.DataFrame( {i: x for i,x in enumerate( zip( *tmp ) ) } )
    df_latlon.columns = ['lat','lon']



    points_ll = np.array( df_latlon[['lon','lat']] )


    ## doing the relaxation rounds 
    data = points_ll

    for r in range(1,nb_rounds+1):


        data = spherical_relax_round( data , nb_sectors=nb_sectors, learning_rate=learning_rate , point_weights=point_weights ) 
        if r % write_every == 0 :
            ## output
            data_xyz = latlon_to_xyz(data[:,1], data[:,0]) ## NB: data is lon,lat ... 
            D = {'x': data_xyz[0],'y': data_xyz[1],'z': data_xyz[2] }
            D.update({c : df_pts[c] for c in additional_columns})
            df = pd.DataFrame( D , index = df_pts.index )
            
            df.to_csv(f"{args.output_prefix}_round{r}.csv", index=True)

