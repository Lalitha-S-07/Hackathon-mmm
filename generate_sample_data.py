import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_data(num_students=1000, output_dir='data'):
    """
    Generate sample student data for all four verification stages.
    
    Args:
        num_students (int): Number of students to generate data for.
        output_dir (str): Directory to save the generated data files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate student IDs
    student_ids = [f"STU_{i:05d}" for i in range(1, num_students + 1)]
    
    # Generate demographic data (Stage 1)
    stage1_data = {
        'student_id': student_ids,
        'age': np.random.randint(16, 25, num_students),
        'gender': np.random.choice(['Male', 'Female', 'Other'], num_students, p=[0.48, 0.48, 0.04]),
        'family_income': np.random.choice(['Low', 'Medium', 'High'], num_students, p=[0.4, 0.4, 0.2]),
        'parent_education': np.random.choice(['Primary', 'Secondary', 'Graduate', 'Postgraduate'], 
                                           num_students, p=[0.2, 0.3, 0.35, 0.15]),
        'family_size': np.random.randint(2, 8, num_students),
        'rural_urban': np.random.choice(['Rural', 'Urban'], num_students, p=[0.6, 0.4]),
        'first_generation': np.random.choice([0, 1], num_students, p=[0.7, 0.3])
    }
    
    # Generate academic data (Stage 2)
    stage2_data = {
        'student_id': student_ids,
        'previous_grades': np.random.normal(70, 15, num_students).clip(0, 100),
        'attendance_rate': np.random.normal(80, 15, num_students).clip(30, 100),
        'extracurricular_activities': np.random.randint(0, 5, num_students),
        'study_hours_per_week': np.random.randint(1, 40, num_students),
        'subject_preferences': np.random.choice(['Science', 'Arts', 'Commerce', 'Vocational'], 
                                              num_students, p=[0.4, 0.2, 0.3, 0.1]),
        'learning_disability': np.random.choice([0, 1], num_students, p=[0.9, 0.1]),
        'special_education_needs': np.random.choice([0, 1], num_students, p=[0.95, 0.05])
    }
    
    # Generate financial data (Stage 3)
    stage3_data = {
        'student_id': student_ids,
        'scholarship_received': np.random.choice([0, 1], num_students, p=[0.6, 0.4]),
        'scholarship_amount': np.where(
            np.random.choice([0, 1], num_students, p=[0.6, 0.4]) == 1,
            np.random.randint(1000, 10000, num_students),
            0
        ),
        'financial_aid': np.random.choice([0, 1], num_students, p=[0.7, 0.3]),
        'work_study_program': np.random.choice([0, 1], num_students, p=[0.8, 0.2]),
        'part_time_job': np.random.choice([0, 1], num_students, p=[0.75, 0.25]),
        'family_financial_support': np.random.choice(['Low', 'Medium', 'High'], num_students, p=[0.3, 0.5, 0.2]),
        'debt_amount': np.random.randint(0, 50000, num_students)
    }
    
    # Generate interview and behavioral data (Stage 4)
    stage4_data = {
        'student_id': student_ids,
        'interview_score': np.random.normal(7, 2, num_students).clip(0, 10),
        'motivation_level': np.random.choice(['Low', 'Medium', 'High'], num_students, p=[0.2, 0.5, 0.3]),
        'career_goals_clarity': np.random.randint(1, 10, num_students),
        'emotional_stability': np.random.randint(1, 10, num_students),
        'peer_relationships': np.random.choice(['Poor', 'Average', 'Good'], num_students, p=[0.15, 0.5, 0.35]),
        'teacher_relationships': np.random.choice(['Poor', 'Average', 'Good'], num_students, p=[0.1, 0.4, 0.5]),
        'disciplinary_issues': np.random.randint(0, 10, num_students)
    }
    
    # Generate dropout status (target variable)
    # Create a weighted combination of factors that influence dropout
    dropout_factors = (
        # Academic factors (30% weight)
        (stage2_data['previous_grades'] < 60) * 0.15 +
        (stage2_data['attendance_rate'] < 70) * 0.15 +
        
        # Financial factors (30% weight)
        (stage1_data['family_income'] == 'Low') * 0.1 +
        (stage3_data['scholarship_received'] == 0) * 0.1 +
        (stage3_data['financial_aid'] == 0) * 0.1 +
        
        # Personal factors (25% weight)
        (stage4_data['motivation_level'] == 'Low') * 0.1 +
        (stage4_data['career_goals_clarity'] < 5) * 0.1 +
        (stage4_data['emotional_stability'] < 5) * 0.05 +
        
        # Social factors (15% weight)
        (stage4_data['peer_relationships'] == 'Poor') * 0.05 +
        (stage4_data['teacher_relationships'] == 'Poor') * 0.05 +
        (stage4_data['disciplinary_issues'] > 5) * 0.05
    )
    
    # Add some randomness
    dropout_factors += np.random.normal(0, 0.1, num_students)
    
    # Convert to binary outcome (1 = dropout, 0 = no dropout)
    dropout_status = (dropout_factors > 0.5).astype(int)
    
    # Add dropout status to stage 4 data
    stage4_data['dropout_status'] = dropout_status
    
    # Create dataframes
    stage1_df = pd.DataFrame(stage1_data)
    stage2_df = pd.DataFrame(stage2_data)
    stage3_df = pd.DataFrame(stage3_data)
    stage4_df = pd.DataFrame(stage4_data)
    
    # Save to CSV files
    stage1_df.to_csv(os.path.join(output_dir, 'stage1_data.csv'), index=False)
    stage2_df.to_csv(os.path.join(output_dir, 'stage2_data.csv'), index=False)
    stage3_df.to_csv(os.path.join(output_dir, 'stage3_data.csv'), index=False)
    stage4_df.to_csv(os.path.join(output_dir, 'stage4_data.csv'), index=False)
    
    # Also create a combined dataset
    combined_df = stage1_df.merge(stage2_df, on='student_id')
    combined_df = combined_df.merge(stage3_df, on='student_id')
    combined_df = combined_df.merge(stage4_df, on='student_id')
    combined_df.to_csv(os.path.join(output_dir, 'combined_data.csv'), index=False)
    
    print(f"Generated sample data for {num_students} students.")
    print(f"Data saved to {output_dir} directory.")
    print(f"Dropout rate: {dropout_status.mean() * 100:.1f}%")
    
    return {
        'stage1': stage1_df,
        'stage2': stage2_df,
        'stage3': stage3_df,
        'stage4': stage4_df,
        'combined': combined_df
    }

if __name__ == "__main__":
    # Generate sample data
    data = generate_sample_data(num_students=1000)