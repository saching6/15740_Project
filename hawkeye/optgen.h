//Hawkeye Cache Replacement Tool v2.0
//UT AUSTIN RESEARCH LICENSE (SOURCE CODE)
//The University of Texas at Austin has developed certain software and documentation that it desires to
//make available without charge to anyone for academic, research, experimental or personal use.
//This license is designed to guarantee freedom to use the software for these purposes. If you wish to
//distribute or make other use of the software, you may purchase a license to do so from the University of
//Texas.
///////////////////////////////////////////////
//                                            //
//     Hawkeye [Jain and Lin, ISCA' 16]       //
//     Akanksha Jain, akanksha@cs.utexas.edu  //
//                                            //
///////////////////////////////////////////////


#ifndef OPTGEN_H
#define OPTGEN_H

using namespace std;

#include <iostream>

#include <math.h>
#include <set>
#include <vector>


// Sachin: address info structure with 8byte address
struct ADDR_INFO
{
    uint64_t addr;
    uint32_t last_quanta;
    uint64_t PC; 
    bool prefetched;
    uint32_t lru;

    void init(unsigned int curr_quanta)
    {
        last_quanta = 0;
        PC = 0;
        prefetched = false;
        lru = 0;
    }

    void update(unsigned int curr_quanta, uint64_t _pc, bool prediction)
    {
        last_quanta = curr_quanta;
        PC = _pc;
    }

    void mark_prefetch()
    {
        prefetched = true;
    }
};

// Make optgen more powerful : 
// 1) Increase OPTGEN_VECTOR_SIZE

struct OPTgen
{
    vector<unsigned int> liveness_history;

    uint64_t num_cache;
    uint64_t num_dont_cache;
    uint64_t access;

    uint64_t CACHE_SIZE;

    void init(uint64_t size)
    {
        num_cache = 0;
        num_dont_cache = 0;
        access = 0;
        CACHE_SIZE = size;
        liveness_history.resize(OPTGEN_VECTOR_SIZE, 0);
    }

    void add_access(uint64_t curr_quanta)
    {
        access++;
        liveness_history[curr_quanta] = 0;
    }

    void add_prefetch(uint64_t curr_quanta)
    {
        liveness_history[curr_quanta] = 0;
    }

    /*
    void PrintShouldCache( uint64_t curr_quanta, uint64_t last_quanta )
    {
	   // printf( "Optgen Should Cache --> \n %u, %u\n", curr_quanta, last_quanta );
    }
    */

    bool should_cache(uint64_t curr_quanta, uint64_t last_quanta)
    {
	//PrintShouldCache( curr_quanta, last_quanta );

        bool is_cache = true;

        unsigned int i = last_quanta;
        while (i != curr_quanta)
        {
            if(liveness_history[i] >= CACHE_SIZE)
            {
                is_cache = false;
                break;
            }

            i = (i+1) % liveness_history.size();
        }


        //if ((is_cache) && (last_quanta != curr_quanta))
        if ((is_cache))
        {
            i = last_quanta;
            while (i != curr_quanta)
            {
                liveness_history[i]++;
                i = (i+1) % liveness_history.size();
            }
            assert(i == curr_quanta);
        }

        if (is_cache) num_cache++;
        else num_dont_cache++;

        return is_cache;    
    }

    uint64_t get_num_opt_hits()
    {
        return num_cache;

        uint64_t num_opt_misses = access - num_cache;
        return num_opt_misses;
    }
};

#endif
