////////////////////////////////////////////
//                                        //
//        LRU replacement policy          //
//     Jinchun Kim, cienlux@tamu.edu      //
//                                        //
////////////////////////////////////////////

#include "../inc/champsim_crc2.h"
#include <stdlib.h>
#include <torch/torch.h>
#include <torch/script.h>

#include <memory>

#include <map>
#include <valarray>

#define NUM_CORE 1
#define LLC_SETS NUM_CORE*2048
#define LLC_WAYS 16

#define PC_WIN_SZ 32
map<int, map<int, int>> set_map;
map<int, valarray<int>> pc_map;

#define MAX_RRPV 7
uint32_t rrpv[LLC_SETS][LLC_WAYS];
valarray<int> set_occ[LLC_SETS];

//Initialize Learned Pytorch Model
torch::jit::script::Module get_module( const char* path ){

    torch::jit::script::Module module;
    try {
                // Deserialize the ScriptModule from a file usingn torch::jit::load().
                module = torch::jit::load( path );
    } 
    catch ( const c10::Error& e ){
        std::cerr << "error loading the model\n";
    }

    std::cout << "Model loaded without issue\n";

    return module;
}

//pytorch traced model path
const char* module_path = "/home/josh/Desktop/Project/15740/15740_Project/dl_cache/traced_GRAPH-TRANSFORMER_BSZ.64_LR.0.0001_saved_model.pth";

//model to use for predictions
torch::jit::script::Module MODULE = get_module( module_path );

// initialize replacement state
// [ldery] - in total, we have LLC_SETS, each with LLC_WAYS. We would like to come up with a "within-set"
// [ldery] replacement policy
void InitReplacementState()
{
    cout << "Initialize RRPV replacement state" << endl;
    
    //initialize pc_maps to valarrays
    valarray<int> valarray_init (PC_WIN_SZ);

    for (int i=0; i<LLC_SETS; i++) {
        for (int j=0; j<LLC_WAYS; j++) {
            rrpv[i][j] = MAX_RRPV; // cache is empty, everything is averse
        }

        //pc_maps initialize to zero arrays of PC_WIN_SZ
        //set occupancies initialized to 0
        pc_map.insert( std::pair<int,valarray<int>>( i, valarray_init ) );
        set_occ[i] = valarray_init;
        for (int k = 0; k < PC_WIN_SZ; k++ ){
            pc_map[i][k] = 0;
	    set_occ[i][k] = 0;
        }
    }

}


// find replacement victim
// return value should be 0 ~ 15 or 16 (bypass)
// [ldery]
//      uint32_t cpu = which core / cpu this is running on (?)
//      uint32_t set = which set in an N-way associative cache this
//      const BLOCK *current_set = Not sure what this is (?)
//      uint64_t PC  = program counter
//      uint64_t paddr = page to replace
//      uint32_t type  =
uint32_t GetVictimInSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type)
{
    for ( uint32_t i = 0; i < LLC_WAYS; i++ )
        if( rrpv[set][i] == MAX_RRPV )
            return i;

    uint32_t max_rrip = 0;
    int32_t lru_victim = -1;
    for ( uint32_t i = 0; i < LLC_WAYS; i++ ){
        if ( rrpv[set][i] >= max_rrip ){
            max_rrip = rrpv[set][i];
            lru_victim = i;
        }
    }

    assert( lru_victim != -1 );
    return lru_victim;
}   

// called on every cache hit and cache fill
// [ldery]
//      uint64_t victim_addr = victim address
//      uint8_t hit = whether it was a hit or a miss

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
    set_occ[ set ][ slice( 0, PC_WIN_SZ - 2, 1 ) ] = set_occ[ set ][ slice( 1, PC_WIN_SZ - 1, 1 ) ];
    if( set_occ[ set ][ PC_WIN_SZ - 1 ] < LLC_WAYS - 1 && !hit )
    	set_occ[ set ][ PC_WIN_SZ - 1 ] = set_occ[ set ][ PC_WIN_SZ - 2 ] + 1;

    if( set_map[set].find( PC ) == set_map[set].end() )
        set_map[ set ][ PC ] = set_map[set].size() + 1;
    
    pc_map[set][ slice( 0, PC_WIN_SZ - 2, 1 ) ] = pc_map[set][ slice( 1, PC_WIN_SZ - 1, 1 ) ];
    pc_map[set][ PC_WIN_SZ - 1 ] = set_map[ set ][ PC ];
    
    std::array<std::valarray<int>, 3> input = { pc_map[set], set_occ[set], set_occ[set] };
    std::vector<torch::jit::IValue> inputs;
    
    inputs.push_back( torch::from_blob( &input, {PC_WIN_SZ, 3} ) );
    cout << inputs << endl;

    at::Tensor output = MODULE.forward( inputs ).toTensor();
    torch::Tensor friendly = output.argmax(1).to(torch::kInt32);
	
    int* friendly_bool = friendly.data_ptr<int>();
    bool bel_friendly = (bool)friendly_bool[ PC_WIN_SZ ];

    //assume module.forward( x ) => new_prediction until its implemented
    if( !bel_friendly )
        rrpv[set][way] = MAX_RRPV;
    else{
        rrpv[set][way] = 0;
        if(!hit){
            bool saturated = false;
            for (uint32_t i=0; i < LLC_WAYS; i++ )
                if (rrpv[set][i] == MAX_RRPV-1)
                    saturated = true;

            //Age all the cache-friendly lines
            for( uint32_t i=0; i < LLC_WAYS; i++ ){
                if ( !saturated && rrpv[set][i] < MAX_RRPV - 1 )
                    rrpv[set][i]++;
            }
        }

        rrpv[set][way] = 0;
    }

}

// use this function to print out your own stats on every heartbeat 
void PrintStats_Heartbeat()
{

}

// use this function to print out your own stats at the end of simulation
void PrintStats()
{

}
