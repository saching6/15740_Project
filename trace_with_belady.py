import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse

def belady( df ):
    sets = 2048
    ways = 16

    reuse_dict = {}

    cache = -1 * np.ones( ( sets, ways ), dtype=np.float128 )
    cache_hist = -1 * np.ones( ( df.shape[0], ways ), dtype=np.float128 )
    reuse_dist = np.zeros( ( df.shape[0], ways ) )

    hit = np.zeros( df.shape[0] )
    friendly = np.zeros( df.shape[0] )
    evict_addr = -1 * np.ones( df.shape[0], dtype=np.float128 )

    for i in tqdm( range( df.shape[0] ) ):
        which_set = df['Set'][i]
        paddr = df['Physical Address'][i]

        if paddr in cache[which_set]:
            hit[i] = 1
        else:
            bel = np.zeros( ways )
            for j, way in enumerate( cache[which_set] ):
                if way < 0:
                    #If the cache hasn't been filled, 
                    #put it in the cache and mark it friendly

                    cache[which_set,j] = paddr
                    friendly[i] = 1
 
                    way_reuse = np.where( df['Physical Address'][i+1:] == paddr )[0]
                    
                    if len( way_reuse ) == 0:
                        bel[j] = np.inf
                    else:
                        bel[j] = way_reuse[0]
                        reuse_dict[paddr] = way_reuse[1:]

                    break
                """
                if way in reuse_dict:
                    way_reuse = reuse_dict.get( way )
                    if len( way_reuse ) > 0:
                        way_reuse -= way_reuse[0]
                else:
                """
                way_reuse = np.where( df['Physical Address'][i+1:] == way )[0]
                
                if len( way_reuse ) == 0:
                    bel[j] = np.inf
                else:
                    bel[j] = way_reuse[0]
                    reuse_dict[way] = way_reuse[1:]

            reuse_dist[i] = bel
            if all( bel > 0 ):
                """
                if paddr in reuse_dict:
                    paddr_reuse = reuse_dict.get( paddr )
                    if len( paddr_reuse ) > 0:
                        paddr_reuse -= paddr_reuse[0]
                else:
                """
                paddr_reuse = np.where( df['Physical Address'][i+1:] == paddr )[0]

                if len( paddr_reuse ) == 0:
                    next_reuse = np.inf
                else:
                     next_reuse = paddr_reuse[0]
                     reuse_dict[paddr] = paddr_reuse[1:]

                if any( next_reuse <= bel ) and next_reuse < np.inf:
                    if ( next_reuse <= bel ).sum() >= 2:
                        #If paddr is at least the second furthest, 
                        #mark it friendly
                        friendly[i] = 1

                    replace = np.argmax( bel )
                    evict_addr[i] = cache[which_set,replace]
                    bel[replace] = next_reuse
                    reuse_dist[i,replace] = next_reuse
                    cache[which_set,replace] = paddr
        
            reuse_dist[i,:] = bel.tolist()
            cache_hist[i,:] = cache[which_set].tolist()
    
    cache_hist = np.array( cache_hist )
    cache_col_names = [ "Cache_Set_{}".format( i ) for i in range( 16 ) ]
    
    cache_df = pd.DataFrame()
    for l, col in enumerate( cache_col_names ):
        cache_df.insert( l, col, cache_hist[:,l] )

    belady_evict = pd.DataFrame( evict_addr, columns=['Belady Evict'] )
    belady_hit = pd.DataFrame( hit, columns=['Belady Hit'] )
    cache_friendly = pd.DataFrame( friendly, columns=["Belady Friendly"] )
    cache_reuse_dist = pd.DataFrame( reuse_dist, 
            columns=['Reuse Distance {}'.format( i ) for i in range( ways) ] )
    
    belady_df = pd.concat(
            [ 
                belady_evict, 
                belady_hit, 
                cache_friendly, 
                cache_df, 
                cache_reuse_dist 
            ], axis=1 )

    return belady_df

def main( ):
    parser = argparse.ArgumentParser(description='Input and output csv for finding optimal')
    parser.add_argument('-i', type=str, help='input_filename' )
    parser.add_argument('-o', type=str, help='output_filename' )

    args = parser.parse_args()
    file_in = args.i
    file_out = args.o

    with open( file_in, 'r+' ) as f:
        trace = pd.read_csv( f )

    belady_df = belady( trace )
    out_trace = pd.concat( [ trace, belady_df ], axis=1 )

    with open( file_out, 'w+' ) as f:
        out_trace.to_csv( f )

if __name__ == "__main__":
    main()

    
