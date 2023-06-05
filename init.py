from model_effectivness import avg_precision, dcg, precision12
from model_implementation import implement_models, root_folder
from t_test import t_test

if __name__ == "__main__":
    # Defining stop words
    stopwords_file = open(f'{root_folder}common-english-words.txt', 'r') 
    stop_words = stopwords_file.read().split(',')
    stopwords_file.close()
    # Implement all 3 models
    implement_models(f"{root_folder}Datasets", f"{root_folder}Queries.txt", stop_words)
    ############### Precision measures ##################
    # MAP
    avg_prec_baseline = avg_precision(f"{root_folder}Feedback", f"{root_folder}Result\Baseline", "Baseline_R*Ranking.dat", f"{root_folder}Result\Precision\Baseline_AP.txt")
    avg_prec_likelihood = avg_precision(f"{root_folder}Feedback", f"{root_folder}Result\Likelihood_IR", "likelihood_R*Ranking.dat", f"{root_folder}Result\Precision\Likelihood_AP.txt")
    avg_prec_vsm = avg_precision(f"{root_folder}Feedback", f"{root_folder}Result/vsmbm25", "VSMBM25_R*Ranking.dat", f"{root_folder}Result\Precision\VSM_AP.txt")
    # Precision@12
    p12_baseline = precision12(f"{root_folder}Feedback", f"{root_folder}Result\Baseline", "Baseline_R*Ranking.dat", f"{root_folder}Result\Precision\Baseline_P12.txt")
    p12_likelihood = precision12(f"{root_folder}Feedback", f"{root_folder}Result\Likelihood_IR", "likelihood_R*Ranking.dat", f"{root_folder}Result\Precision\Likelihood_P12.txt")
    p12_vsm = precision12(f"{root_folder}Feedback", f"{root_folder}Result/vsmbm25", "VSMBM25_R*Ranking.dat", f"{root_folder}Result\Precision\VSM_P12.txt")
    # DCG
    dcg_baseline = dcg(f"{root_folder}Feedback", f"{root_folder}Result\Baseline", "Baseline_R*Ranking.dat", f"{root_folder}Result\Precision\Baseline_dcg.txt")
    dcg_likelihood = dcg(f"{root_folder}Feedback", f"{root_folder}Result\Likelihood_IR", "likelihood_R*Ranking.dat", f"{root_folder}Result\Precision\Likelihood_dcg.txt")
    dcg_vsm = dcg(f"{root_folder}Feedback", f"{root_folder}Result/vsmbm25", "VSMBM25_R*Ranking.dat", f"{root_folder}Result\Precision\VSM_dcg.txt")
    print("DGC_likelihood", dcg_likelihood)
    ################### T- test ############################
    # MAP
    print("-------------------------------------------------------------------------------------------------------")
    print("T-test results based on Average Precision:")
    print("-------------------------------------------------------------------------------------------------------")

    print("T-test results for Baseline vs Likelihood model based on Average Precision: ")
    print(t_test(avg_prec_baseline, avg_prec_likelihood))
    print("T-test results for Baseline vs Vector Space model based on Average Precision: ")
    print(t_test(avg_prec_baseline, avg_prec_vsm))
    print("T-test results for Likelihood vs Vector Space model based on Average Precision: ")
    print(t_test(avg_prec_likelihood, avg_prec_vsm))
    # Precision@12
    print("-------------------------------------------------------------------------------------------------------")
    print("T-test results based on Precision@12:")
    print("-------------------------------------------------------------------------------------------------------")

    print("T-test results for Baseline vs Likelihood model based on Precision@12: ")
    print(t_test(p12_baseline, p12_likelihood))
    print("T-test results for Baseline vs Vector Space model based on Precision@12: ")
    print(t_test(p12_baseline, p12_vsm))
    print("T-test results for Likelihood vs Vector Space model based on Precision@12: ")
    print(t_test(p12_likelihood, p12_vsm))
    # DCG
    print("-------------------------------------------------------------------------------------------------------")
    print("T-test results based on DCG:")
    print("-------------------------------------------------------------------------------------------------------")

    print("T-test results for Baseline vs Likelihood model based on DCG: ")
    print(t_test(dcg_baseline, dcg_likelihood))
    print("T-test results for Baseline vs Vector Space model based on DCG: ")
    print(t_test(dcg_baseline, dcg_vsm))
    print("T-test results for Likelihood vs Vector Space model based on DCG: ")
    print(t_test(dcg_likelihood, dcg_vsm))
