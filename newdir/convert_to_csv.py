import re
import argparse

def clean( file_in, file_out ):
    re_pattern = "\d{1,20}, "*7
    re_pattern += "\d{1,20}"

    with open( file_in, "r+" ) as f:
        lines = f.readlines()

    with open( file_out, "w+" ) as f:
        f.write( "Set,Way,Physical Address,Victim Address,Program Counter,Type,Hit,Cache Friendly" )
        for line in lines:
            val = re.search( re_pattern, line )
            if val is not None:
                dataline = val.group(0)
                dataline = dataline.replace( " ", "" )
 
                f.write("\n")
                f.write( dataline )

def main( ):
    parser = argparse.ArgumentParser(description="Clean fprint raw output")
    parser.add_argument( "-i", type=str, help="the input filename" )
    parser.add_argument( "-o", type=str, help="the output filename" )

    args = parser.parse_args()
    file_in = args.i
    file_out = args.o

    clean( file_in, file_out )

main()
