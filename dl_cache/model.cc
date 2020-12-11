#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <memory>

torch::jit::script::Module get_module( char path ){

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file usingn torch::jit::load().
		module = torch::jit::load( path );
	}
    catch ( const c10::Error& e ){
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "Model loaded without issue\n";

    return module;

module_path = "traced_GRAPH-TRANSFORMER-BSZ.64-LR.0.0001_saved_model.pth";
torch::jit::script::Module module = get_module( module_path );

