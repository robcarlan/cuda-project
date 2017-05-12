
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

extern int mlmc(bool use_debug, bool use_timings)

int main (int argc, char **argv) {

    int c;

    int debug_flag = 0;
    int use_timings = 0;

    //Option parser
    while (c != -1) {

		static struct option long_options[]  = {
			{"debug", no_argument, &debug_flag, 1},
			{"timings", no_argument, &use_timings, 't'},
			{"file", required_argument, 0, 'f'},
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
			default:
			continue;
		}
    }

    printf("Hi \n");

    //Main MLMC code
    mlmc(debug_flag, use_timings);


    return 0;
}
