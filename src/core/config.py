import numpy as np

class Config:
    VIDEO_PATH = 'data/1/corte_video.mp4'
    STAB_SCALE = 0.4  

    # Parámetros base
    MIN_FRAMES_TO_CONFIRM = 5  
    MIN_AREA = 15        
    MAX_AREA = 2500    
    DIST_THRESH = 350  
    MAX_MISSING = 40   
    MIN_SOLIDITY = 0.70  
    EDGE_MARGIN = 5 

    # Física
    MIN_DISPLACEMENT_BASE = 50  
    MAX_TORTUOSITY = 1.3  
    GRAVEDAD_PIXELS = 1.2 

    # Zonas
    ORIGIN_ZONE = np.array([
        [1939,1519], [2469,844], [2379,726], [2227,627], [2190,606],
        [2155,587], [2093,553], [1864,810], [1810,873], [1648,1062],
        [1622,1110], [1526,1298], [1501,1348], [1565,1376], [1762,1451]
    ], np.int32).reshape((-1, 1, 2))

    EXPECTED_PROJECTION_ZONE = np.array([
        [3948, 901], [3107, 187], [2750, 23], [2680, -5], [2644, -19],
        [1781, -356], [918, 324], [852, 377], [580, 598], [462, 734],
        [325, 897], [-652, 2064], [665, 2860], [942, 3005], [2663, 3898]
    ], np.int32).reshape((-1, 1, 2))

    # Matriz Homografía
    H_MATRIX = np.array([ 
        -0.015931589199440533, 0.08273002461280851, -6764.196562873699, 
        0.048673511102109844, -0.0033261794074875787, -4584.529246474536, 
        -0.000013337377566862186, 0.000003317779833310359, 1 
    ]).reshape(3, 3)