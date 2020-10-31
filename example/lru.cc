////////////////////////////////////////////
//                                        //
//        LRU replacement policy          //
//     Jinchun Kim, cienlux@tamu.edu      //
//                                        //
////////////////////////////////////////////

#include "../inc/champsim_crc2.h"

#define NUM_CORE 1
#define LLC_SETS NUM_CORE*2048
#define LLC_WAYS 16

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
    for (int i=0; i<LLC_WAYS; i++)
        if (lru[set][i] == (LLC_WAYS-1))
            return i;

    return 0;
}

// called on every cache hit and cache fill
// [ldery]
//      uint64_t victim_addr = victim address
//      uint8_t hit = whether it was a hit or a miss

void UpdateReplacementState (uint32_t cpu, uint32_t set, uint32_t way, uint64_t paddr, uint64_t PC, uint64_t victim_addr, uint32_t type, uint8_t hit)
{
    // update lru replacement state
    for (uint32_t i=0; i<LLC_WAYS; i++) {
        if (lru[set][i] < lru[set][way]) {
            lru[set][i]++;

            if (lru[set][i] == LLC_WAYS)
                assert(0);
        }
    }
    lru[set][way] = 0; // promote to the MRU position
}

// use this function to print out your own stats on every heartbeat 
void PrintStats_Heartbeat()
{

}

// use this function to print out your own stats at the end of simulation
void PrintStats()
{

}
