import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_website_optimization(user_data, ab_test_results, survey_responses):
    """
    Comprehensive analysis of website optimization combining A/B test results,
    user behavior data, and customer survey responses.
    """
    
    def perform_ab_test_analysis():
        # Perform statistical analysis of A/B test results
        control_data = ab_test_results[ab_test_results['test_group'] == 'control']['conversion']
        treatment_data = ab_test_results[ab_test_results['test_group'] == 'treatment']['conversion']
        
        # Calculate conversion rates
        control_rate = control_data.mean() * 100
        treatment_rate = treatment_data.mean() * 100
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def analyze_user_behavior():
        # Calculate key metrics
        metrics = {
            'avg_session_duration': user_data['session_duration'].mean(),
            'avg_pages_viewed': user_data['pages_viewed'].mean(),
            'avg_purchase_amount': user_data[user_data['purchase_amount'] > 0]['purchase_amount'].mean(),
            'conversion_rate': (user_data['purchase_amount'] > 0).mean() * 100,
            'bounce_rate': (user_data['pages_viewed'] == 1).mean() * 100
        }
        
        # Segment users by engagement level
        user_data['engagement_score'] = (
            (user_data['session_duration'] / user_data['session_duration'].max()) * 0.4 +
            (user_data['pages_viewed'] / user_data['pages_viewed'].max()) * 0.4 +
            (user_data['purchase_amount'] > 0) * 0.2
        )
        
        return metrics
    
    def analyze_customer_feedback():
        # Only analyze feedback from users who made purchases
        purchased_users = user_data[user_data['purchase_amount'] > 0]['user_id']
        valid_surveys = survey_responses[survey_responses['user_id'].isin(purchased_users)]
        
        satisfaction_metrics = {
            'avg_satisfaction': valid_surveys['satisfaction_score'].mean(),
            'satisfaction_distribution': valid_surveys['satisfaction_score'].value_counts(),
            'top_feedback_categories': valid_surveys['feedback_category'].value_counts().head()
        }
        
        return satisfaction_metrics
    
    # Combine all analyses
    results = {
        'ab_test_results': perform_ab_test_analysis(),
        'user_behavior': analyze_user_behavior(),
        'customer_feedback': analyze_customer_feedback()
    }
    
    return results

def generate_sample_data(n_users=1000):
    np.random.seed(42)
    
    # Generate user behavior data with more realistic distributions
    user_data = pd.DataFrame({
        'user_id': range(n_users),
        'session_duration': np.random.gamma(shape=2, scale=4, size=n_users),  
        'pages_viewed': np.random.negative_binomial(n=3, p=0.3, size=n_users) + 1,  
        'purchase_amount': np.zeros(n_users) 
    })
    
    purchasers = np.random.choice(
        n_users,
        size=int(n_users * 0.025),  
        replace=False
    )
    user_data.loc[purchasers, 'purchase_amount'] = np.random.gamma(
        shape=5,
        scale=20,
        size=len(purchasers)
    )  
    
    ab_test_results = pd.DataFrame({
        'user_id': range(n_users),
        'test_group': np.random.choice(['control', 'treatment'], n_users),
        'conversion': np.random.binomial(
            n=1,
            p=0.025,  
            size=n_users
        )
    })
    
    # Slightly increase conversion rate for treatment group
    treatment_mask = ab_test_results['test_group'] == 'treatment'
    ab_test_results.loc[treatment_mask, 'conversion'] = np.random.binomial(
        n=1,
        p=0.028,  
        size=treatment_mask.sum()
    )
    
    # Generate survey responses only for users who made purchases
    survey_responses = pd.DataFrame({
        'user_id': purchasers,
        'satisfaction_score': np.random.normal(4.2, 0.5, len(purchasers)).clip(1, 5),
        'feedback_category': np.random.choice(
            ['UI/UX', 'Performance', 'Content', 'Pricing', 'Support'],
            len(purchasers),
            p=[0.3, 0.25, 0.2, 0.15, 0.1]  # Weighted distribution of feedback
        )
    })
    
    return user_data, ab_test_results, survey_responses

def print_results(results):
    """Function to print formatted results"""
    print("\nWebsite Optimization Analysis Results\n")
    
    print("1. A/B Test Results:")
    print(f"   - Control Group Conversion Rate: {results['ab_test_results']['control_rate']:.2f}%")
    print(f"   - Treatment Group Conversion Rate: {results['ab_test_results']['treatment_rate']:.2f}%")
    print(f"   - Absolute Improvement: {(results['ab_test_results']['treatment_rate'] - results['ab_test_results']['control_rate']):.2f}%")
    print(f"   - P-value: {results['ab_test_results']['p_value']:.4f}")
    print(f"   - Statistically Significant: {results['ab_test_results']['significant']}")
    
    print("\n2. User Behavior Metrics:")
    print(f"   - Average Session Duration: {results['user_behavior']['avg_session_duration']:.2f} minutes")
    print(f"   - Average Pages Viewed: {results['user_behavior']['avg_pages_viewed']:.2f}")
    print(f"   - Average Purchase Amount: ${results['user_behavior']['avg_purchase_amount']:.2f}")
    print(f"   - Conversion Rate: {results['user_behavior']['conversion_rate']:.2f}%")
    print(f"   - Bounce Rate: {results['user_behavior']['bounce_rate']:.2f}%")
    
    print("\n3. Customer Feedback Analysis:")
    print(f"   - Average Satisfaction Score: {results['customer_feedback']['avg_satisfaction']:.2f}/5.0")
    print("\n   Top Feedback Categories:")
    for category, count in results['customer_feedback']['top_feedback_categories'].items():
        print(f"   - {category}: {count} responses")


def visualize_results(results, user_data):
    """Create simple visualizations of key website metrics"""
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: A/B Test Results
    groups = ['Control', 'Treatment']
    rates = [
        results['ab_test_results']['control_rate'],
        results['ab_test_results']['treatment_rate']
    ]
    ax1.bar(groups, rates)
    ax1.set_title('Conversion Rates')
    ax1.set_ylabel('Conversion Rate (%)')
    
    # Plot 2: Purchase Distribution
    purchasers = user_data[user_data['purchase_amount'] > 0]
    ax2.hist(purchasers['purchase_amount'], bins=20)
    ax2.set_title('Purchase Amounts')
    ax2.set_xlabel('Amount ($)')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Generate sample data
    user_data, ab_test_results, survey_responses = generate_sample_data(20000)
    results = analyze_website_optimization(user_data, ab_test_results, survey_responses)
    print_results(results)
    visualize_results(results, user_data)
