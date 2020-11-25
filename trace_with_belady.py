import pandas as pd
import numpy as np

import argparse

def belady( df ):
    sets = 2048
    ways = 16

    cache = -1 * np.ones( ( sets, ways ), dtype=np.float64 )
    cache_hist = -1 * np.ones( ( df.shape[0], ways ) )
    
    hit = np.zeros( df.shape[0] )
    evict_way = -1 * np.ones( df.shape[0] )

    for i in range( df.shape[0] ):
        which_set = df['Set'][i]
        p_addr = df['Physical Address'][i]

        if p_addr in cache[which_set]:
            hit[i] = 1
			# still need to determine whether friendly or not
        else:
            bel = np.zeros( ways )
            for j, way in enumerate( cache[which_set] ):
                if way < 0:
                    cache[which_set,j] = p_addr
                    break

                future_acc = np.where( df['Physical Address'][i+1:] == way )[0]

                if len( future_acc ) == 0:
                    bel[j] = np.inf
                else:
                    bel[j] = future_acc[0]

			if all( bel > 0 ):
				next_access = np.where( df['Physical Address'][i+1:] == p_addr )[0]
				if len( next_access ) == 0:
					# cache-averse
					next_access = np.inf
				else:
					next_access = next_access[0]

				if any( next_access <= bel ):
					# Approach 1. 
					# why would we want to keep something else if 
					# cache-friendly if it's reuse distance is less than the second worst
					# cache-averse if it's reuse-distance is greater than second worst but less than worse
					# Approach 2
					# [a=10, b=5, c=11, d=6] -> which to evict
					# 
					replace = np.argmax( bel )
					evict_way[i] = replace
					cache[which_set,replace] = p_addr
				else:
					# mark cache-averse ?

                cache_hist[i,:] = cache[which_set].tolist()
    
    cache_hist = np.array( cache_hist )
    cache_col_names = [ "Cache_Set_{}".format( i ) for i in range( 16 ) ]
    
    cache_df = pd.DataFrame()
    for l, col in enumerate( cache_col_names ):
        cache_df.insert( l, col, cache_hist[:,l] )

    belady_evict = pd.DataFrame( evict_way, columns=['Belady Evict'] )
    belady_hit = pd.DataFrame( hit, columns=['Belady Hit'] )

    belady_df = pd.concat( [ belady_evict, belady_hit, cache_df ], axis=1 )
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

    
