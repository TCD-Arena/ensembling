#!/bin/bash

# This runs all experiments for a single dataset of a  assumption violation.
# For the sets with 7 variables, a bigger max lag is used.


cd ..
cd ..
cd cd_zoo

cmd="python run_methods_on_causal_rivers.py save_full_out=True save_predictions=True method="


# Best performing methods for WCG
methods=(
#"direct_crosscorr"
#"varlingam method.prune=True"
#"var method.base_on=coefficients"
#"pcmci ci_test=RobustParCorr"
#"pcmciplus ci_test=RobustParCorr method.reset_lagged_links=False"
#"dynotears method.lambda_w=0.1 method.lambda_a=0.1 method.max_iter=100 method.h_tol=0.00001"
#"ntsnotears method.h_tol=1e-60 method.rho_max=1e+16 method.lambda1=0.005 method.lambda2=0.01"
#"cp method.architecture=transformer"
"fpcmci method.ci_test=robust_parcorr"
#"svarrfci ci_test=RobustParCorr"
)


cmd4=" method.max_lag=3" 





for method in "${methods[@]}"; do


    if [[ "$method" == *"physical "* ||  "$method" == *"cp"* ]]; then
            echo "$cmd$method"
            eval "$cmd$method" &
    else
            echo  "$cmd$method $cmd4" 
            eval "$cmd$method $cmd2" 
    fi
done

wait
echo "Done"