import sys
sys.path.append( ".." )

import argparse
import torch

import numpy as np

from models import Model
from models import get_model
from dataformatter import csv_to_data
from dataformatter import group_by_set

DATA_PATH = "../hawkeye_trace_belady_graph.csv"
print( torch.cuda.is_available() )

def load_model( fname ):
    model = get_model( "TRANSFORMER" )
    chosen_columns = model.get_data_columns()
    dataset = csv_to_data( DATA_PATH, chosen_columns )

    train_setwise_dataset = group_by_set( dataset )
    train_keys = list( train_setwise_dataset.keys() )
    vals = [ len( x ) for x in list( train_setwise_dataset.values() ) ]
    
    max_key = train_keys[ np.argmax( vals ) ]
    model.prep_for_data( train_setwise_dataset[ max_key ], temp_order=True )
    model.use_cuda = torch.cuda.is_available()

    model.load_state_dict(torch.load( fname, map_location=torch.device('cpu')))

    model.eval()
    return model, train_setwise_dataset[max_key][:32]

def main( ):
    parser = argparse.ArgumentParser(description="Select torch model to convert")
    parser.add_argument("-f", "--file", 
            help="Filename", 
            type=str, 
            required=True,
            default=""
        )
    args = parser.parse_args()

    model, example = load_model( args.file )
    example = torch.from_numpy( example )

    traced_script_module = torch.jit.trace( model, example )
    traced_script_module.save( "traced_{}".format( args.file ) )

    #m_out = model.forward( np.zeros( ( 32, 3 ) ) )
    #print( m_out.argmax( dim=-1 ) )
main()
