////////////////////////////////////////////
//                                        //
//        LRU replacement policy          //
//     Jinchun Kim, cienlux@tamu.edu      //
//                                        //
////////////////////////////////////////////

#include "../inc/champsim_crc2.h"
#include <stdlib.h>

#define NUM_CORE 1
#define LLC_SETS NUM_CORE*2048
#define LLC_WAYS 16

//srand((unsigned)0);

/*
void PrintVictimSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type);
void PrintReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit);
*/

uint32_t lru[LLC_SETS][LLC_WAYS];


// initialize replacement state
// [ldery] - in total, we have LLC_SETS, each with LLC_WAYS. We would like to come up with a "within-set"
// [ldery] replacement policy
void InitReplacementState()
{
    cout << "Initialize LRU replacement state" << endl;

    for (int i=0; i<LLC_SETS; i++) {
        for (int j=0; j<LLC_WAYS; j++) {
            lru[i][j] = j; // [ldery] - why do we initialize this to j
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
    return (int)rand() % LLC_WAYS;
}

/*
// print arguments for GetVictimSet
void PrintVictimSet (uint32_t cpu, uint32_t set, const BLOCK *current_set, uint64_t PC, uint64_t paddr, uint32_t type)
{

	printf( "Victim Set --> \n %u, %u, %u, %u, %u, ", cpu, set, PC, paddr, type );
	printf( "%d, %d, %u, %u, %u, %u, %u\n", (*current_set).valid, (*current_set).dirty,
	 		(*current_set).address, (*current_set).full_addr, (*current_set).tag,
	 		(*current_set).data, (*current_set).cpu, (*current_set).lru );
}
*/


// called on every cache hit and cache fill
// [ldery]
//      uint64_t victim_addr = victim address
//      uint8_t hit = whether it was a hit or a miss

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
    //cout<<"--------------------------------UPDATE VICTIM CALLED-----------------------------"<<endl;
    // PrintReplacementState( cpu, set, way, paddr, PC, victim_addr, type, hit );
    printf( "%u, %u, %lu, %lu, %lu, %u, %u, %u\n", set, way, paddr, victim_addr, PC, type, hit, false ); 
}

/*
// print arguments for UpdateReplacementState
void PrintReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
	printf( "Replacement State --> \n %u, %u, %u, %u, %u, %u, %u, %u\n", cpu, set, way, paddr, PC, victim_addr, type, hit );
}
*/

// use this function to print out your own stats on every heartbeat 
void PrintStats_Heartbeat()
{

}

// use this function to print out your own stats at the end of simulation
void PrintStats()
{

}
