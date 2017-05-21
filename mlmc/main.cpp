#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <chrono>
#include <iostream>

float mlmc_cpu(
    int num_levels,
    int n_initial, float epsilon,
    float alpha_0, float beta_0, float gamma_0,
    int *out_samples_per_level, float *out_cost_per_level,
    bool use_debug, bool use_timings);

float mlmc_gpu(
    int num_levels,
    int n_initial, int timesteps, float epsilon, 
    float alpha_0, float beta_0, float gamma_0, 
    int *out_samples_per_level, float *out_cost_per_level,
    int use_debug, bool use_timings, 
	bool gpu_reduce, bool milstein);

float monte_carlo_gpu(
    int num_levels,
    int n_initial, float epsilon,
    float alpha_0, float beta_0, float gamma_0,
    int &out_samples_per_level, float &out_cost_per_level,
    bool use_debug, bool use_timings);

void run_and_print_stats(
    const char * run_name,
    int num_levels,
    int n_initial, int num_timesteps, float epsilon,
    float alpha, float beta, float gamma,
    int debug_level, bool use_timings,
    bool gpu_version, int variation);

int main (int argc, char **argv) {
	
	std::cout << "HI" << std::endl;

    int c;

    int debug_flag = 0;
    int use_timings = 0;
	
    int num_levels = 3;
    int num_initial = 100;
    int num_timesteps = 64;
    
    float epsilon = 0.1f;
    float alpha = -1.0f;
    float beta = -1.0f;
    float gamma = -1.0f;
	
    char *fp_out = 0;

    int use_cpu = 0;
    int use_gpu1 = 0;
    int use_gpu2 = 0;
	 int use_gpu3 = 0;

    //Option parser
    while (c != -1) {

	static struct option long_options[]  = {
	    {"debug", 	required_argument, 0, 'd'},
	    {"num_levels", 	required_argument, 	&num_levels, 'l'},
	    {"num_initial", required_argument, 	0, 'i'},
	    {"num_timesteps", required_argument, 	0, 't'},
	    {"epsilon", 	required_argument, 	0, 'e'},
	    {"alpha", 		required_argument, 	0, 'a'},
	    {"beta", 		required_argument, 	0, 'b'},
	    {"gamma", 		required_argument, 	0, 'g'},
	    {"timings", 	no_argument, 		&use_timings, 1},
	    {"file", 		required_argument, 	0, 'f'},
	    //Which test to run
		{"gpu3", 	no_argument, 		&use_gpu3, 1},
	    {"gpu2", 	no_argument, 		&use_gpu2, 1},
	    {"gpu1", 	no_argument, 		&use_gpu1, 1},
	    {"cpu", 	no_argument, 		&use_cpu, 1},
	    {"all", 	no_argument, 		0, 'z'},
	    {"help", no_argument, 0, 'h'},
	    {0, 0, 0, 0}
	};

	int opt_index = 0;

	c = getopt_long(argc, argv, "f:", long_options, &opt_index);

	if (c == -1) {
	    //Will break on next time round.
	    continue;
	}

	switch(c) {
	case 'h':
	    printf("Usage: \n");
	    printf("--num_levels = number of mlmc levels to use. \n");
	    printf("--num_initial = number of initial samples to use. \n");
	    printf("--num_timesteps = number of teimsteps to use. \n");
	    printf("--epsilon = goal accuracy epsilon. \n");
	    printf("--alpha = Desired alpha parameter. \n");
	    printf("--beta = Desired beta parameter. \n");
	    printf("--gamma = Desired gamma parameter. \n");
	    printf("--timings = Flag to print timings. \n");
	    printf("--debug = Flag to print debug output. \n");

	    return 0;

	case 'f':
	    printf("option -f with %s\n", optarg);
	    fp_out = optarg;
	    break;
	case 'e':
	    printf("using epsilon :%s\n", optarg);
	    epsilon = atof(optarg);
	    break;
	case 'l':
	    printf("using num_levels :%s\n", optarg);
	    num_levels = atoi(optarg);
	    break;
	case 'i':
	    printf("using num_initial :%s\n", optarg);
	    num_initial = atoi(optarg);
	    break;
	case 't':
	    printf("using num_timesteps :%s\n", optarg);
	    num_timesteps = atoi(optarg);
	    break;
	case 'a':
	    printf("using alpha :%s\n", optarg);
	    alpha = atof(optarg);
	    break;
	case 'b':
	    printf("using beta :%s\n", optarg);
	    beta = atof(optarg);
	    break;
	case 'g':
	    printf("using gamma :%s\n", optarg);
	    gamma = atof(optarg);
	    break;
	case 'd':
	    printf("Using debug level: %d \n", atoi(optarg));
	    debug_flag = atoi(optarg);
	    break;
	case 'z':
	    printf("Running all tests\n");
	    use_cpu = use_gpu1 = use_gpu2 = 1;
	default:
	    continue;
	}
    }

    bool gpu_version = false;

    if (use_cpu)
		run_and_print_stats(
			"CPU",
			num_levels, num_initial, num_timesteps,
			epsilon,
			alpha, beta, gamma,
			debug_flag, use_timings,
			gpu_version, 0);

    gpu_version = true;

    if (use_gpu1)
	run_and_print_stats(
	    "GPU Euler ",
	    num_levels, num_initial, num_timesteps,
	    epsilon,
	    alpha, beta, gamma,
	    debug_flag, use_timings,
	    gpu_version, 0);
		
    if (use_gpu2)
	run_and_print_stats(
	    "GPU_Reduce Euler ",
	    num_levels, num_initial, num_timesteps,
	    epsilon,
	    alpha, beta, gamma,
	    debug_flag, use_timings,
	    gpu_version, 1);


    gpu_version = true;

    if (use_gpu3)
	run_and_print_stats(
	    "GPU_Milstein no GPU Reduce ",
	    num_levels, num_initial, num_timesteps,
	    epsilon,
	    alpha, beta, gamma,
	    debug_flag, use_timings,
	    gpu_version, 2);

    return 0;
}

void run_and_print_stats(
    const char * run_name,
    int num_levels,
    int n_initial, int num_timesteps, float epsilon,
    float alpha, float beta, float gamma,
    int  debug_level, bool use_timings,
    bool gpu_version, int variation) {
		
	using namespace std;
	using namespace std::chrono;

    printf("----------------------------\n");
    printf("Running %s: \n", run_name);
    printf("----------------------------\n");
    int *p_samples_per_level_out = (int *)malloc((num_levels+1)*sizeof(int));
    float *p_cost_per_level_out = (float *)malloc((num_levels+1)*sizeof(float));

    float val;

    printf("num_levels: %d\nnum_initial: %d\nnum_timesteps: %d\nepsilon: %f\nalpha: %f\nbeta: %f\ngamma: %f\n", 
	   num_levels, n_initial, num_timesteps, epsilon, alpha, beta, gamma);
	   
	high_resolution_clock::time_point t1 = high_resolution_clock::now();

    if (!gpu_version) {
    	//Run the CPU version this project is based off
	val = mlmc_cpu(
	    num_levels, n_initial, epsilon,
	    alpha, beta, gamma,
	    p_samples_per_level_out,
	    p_cost_per_level_out,
	    debug_level, use_timings);
    } else {
    	//Run the GPU version. Switch based on variant.
	bool gpu_reduce = false;
	bool use_milstein = false;
	
	if (variation == 1) gpu_reduce = true;
	if (variation == 2) use_milstein = true;
	
	val = mlmc_gpu(
	    num_levels, n_initial, num_timesteps,
	    epsilon,
	    alpha, beta, gamma,
	    p_samples_per_level_out,
	    p_cost_per_level_out,
	    debug_level, use_timings, 
	    gpu_reduce, use_milstein);
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    for (int i = 0; i < num_levels; i++) {
    	printf("Level %d: Num samples - %d, Cost - %f \n", i,
	      p_samples_per_level_out[i],  p_cost_per_level_out[i]);
    }
	
	printf("Estimated  value: %f \n", val);
	
	if (use_timings) {
		std::chrono::duration<float, std::milli> fp_ms = t2 - t1;
		printf("Ran in %.2f ms\n", fp_ms.count());
    }

}
