

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

extern float mlmc(
	int num_levels,
	int n_initial, float epsilon, 
	float alpha_0, float beta_0, float gamma_0, 
	int *out_samples_per_level, float *out_cost_per_level,
	bool use_debug, bool use_timings);

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
			{"debug", 		no_argument, &debug_flag, 1},
			{"num_levels", 	required_argument, 	&num_levels, 'l'},
			{"num_initial", required_argument, 	&num_initial, 'i'},
			{"epsilon", 	required_argument, 	0, 'e'},
			{"alpha", 		required_argument, 	0, 'a'},
			{"beta", 		required_argument, 	0, 'b'},
			{"gamma", 		required_argument, 	0, 'g'},
			{"timings", 	no_argument, 		&use_timings, 1},
			{"file", 		required_argument, 	0, 'f'},
			{0, 0, 0, 0}
		};

		int opt_index = 0;

		c = getopt_long(argc, argv, "f:", long_options, &opt_index);

		if (c == -1) {
			//Will break on next time round.
			continue;
		}

		switch(c) {
			case 'f':
				printf("option -f with %s", optarg);
				fp_out = optarg;
			case 'e':
				printf("using epsilon :%s", optarg);
				epsilon = atof(optarg);
			case 'l':
				printf("using num_levels :%s", optarg);
				num_levels = atoi(optarg);	
			case 'i':
				printf("using num_initial :%s", optarg);
				num_initial = atoi(optarg);	
			case 'a':
				printf("using alpha :%s", optarg);
				alpha = atoi(optarg);	
			case 'b':
				printf("using beta :%s", optarg);
				beta = atoi(optarg);	
			case 'g':
				printf("using gamma :%s", optarg);
				gamma = atoi(optarg);	
			default:
				continue;
		}
    }

    printf("Hi \n");

    int *p_samples_per_level_out;
    float *p_cost_per_level_out;

    //Main MLMC code
    float val = mlmc(
	num_levels, num_initial, epsilon,
	alpha, beta, gamma,
	p_samples_per_level_out,
	p_cost_per_level_out,
	debug_flag, use_timings);


    return 0;
}
