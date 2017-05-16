

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

extern float mlmc_cpu(
	int num_levels,
	int n_initial, float epsilon,
	float alpha_0, float beta_0, float gamma_0,
	int *out_samples_per_level, float *out_cost_per_level,
	bool use_debug, bool use_timings);

extern float mlmc_gpu(
	int num_levels,
	int n_initial, float epsilon, 
	float alpha_0, float beta_0, float gamma_0, 
	int *out_samples_per_level, float *out_cost_per_level,
	bool use_debug, bool use_timings);

extern int monte_carlo_gpu(
	int num_levels,
	int n_initial, float epsilon,
	float alpha_0, float beta_0, float gamma_0,
	int &out_samples_per_level, float &out_cost_per_level,
	bool use_debug, bool use_timings);

void run_and_print_stats(const char * run_name,
	int num_levels,
	int n_initial, float epsilon,
	float alpha, float beta, float gamma,
	bool use_debug, bool use_timings,
	bool gpu_version, int variation);

int main (int argc, char **argv) {

    int c;

    int debug_flag = 0;
    int use_timings = 0;
	
    int num_levels = 3;
    int num_initial = 100;
	
    float epsilon = 0.1f;
    float alpha = -1.0f;
    float beta = -1.0f;
    float gamma = -1.0f;
	
    char *fp_out = 0;

    //Option parser
    while (c != -1) {

		static struct option long_options[]  = {
			{"debug", 		no_argument, 0, 'd'},
			{"num_levels", 	required_argument, 	&num_levels, 'l'},
			{"num_initial", required_argument, 	0, 'i'},
			{"epsilon", 	required_argument, 	0, 'e'},
			{"alpha", 		required_argument, 	0, 'a'},
			{"beta", 		required_argument, 	0, 'b'},
			{"gamma", 		required_argument, 	0, 'g'},
			{"timings", 	no_argument, 		&use_timings, 1},
			{"file", 		required_argument, 	0, 'f'},
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
				printf("Using debug mode. \n");
				debug_flag = 1;
				break;
		    default:
		    	continue;
		}
    }

    bool gpu_version = false;

    run_and_print_stats("CPU",
		num_levels, num_initial, epsilon,
		alpha, beta, gamma,
		debug_flag, use_timings,
		gpu_version, 0);

    gpu_version = true;

    run_and_print_stats("GPU",
		num_levels, num_initial, epsilon,
		alpha, beta, gamma,
		debug_flag, use_timings,
		gpu_version, 0);

    return 0;
}

void run_and_print_stats(
		const char * run_name,
		int num_levels,
		int n_initial, float epsilon,
		float alpha, float beta, float gamma,
		bool use_debug, bool use_timings,
		bool gpu_version, int variation) {

    printf("----------------------------\n");
    printf("Running %s: \n", run_name);
    printf("----------------------------\n");
    int *p_samples_per_level_out = (int *)malloc((num_levels+1)*sizeof(int));
    float *p_cost_per_level_out = (float *)malloc((num_levels+1)*sizeof(float));

    float val;

    printf("num_levels: %d\nnum_initial: %d\nepsilon: %f\nalpha: %f\nbeta: %f\ngamma: %f", 
	   num_levels, n_initial, epsilon, alpha, beta, gamma);

    if (!gpu_version) {
    	//Run the CPU version this project is based off
    	 val = mlmc_cpu(
			num_levels, n_initial, epsilon,
			alpha, beta, gamma,
			p_samples_per_level_out,
			p_cost_per_level_out,
			use_debug, use_timings);
    } else {
    	//Run the GPU version. Switch based on variant.
		 val = mlmc_gpu(
				num_levels, n_initial, epsilon,
				alpha, beta, gamma,
				p_samples_per_level_out,
				p_cost_per_level_out,
				use_debug, use_timings);
    }

    for (int i = 0; i < num_levels; i++) {
    	printf("Level %d: Num samples - %d, Cost - %f \n", i,
    			p_cost_per_level_out[i], p_samples_per_level_out[i]);
    }

}
