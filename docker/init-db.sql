-- Database initialization script for Spin-Torque RL-Gym
-- Creates necessary tables for experiment tracking and analytics

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    tags TEXT[],
    
    CONSTRAINT experiments_status_check CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    step INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    artifact_name VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(100) NOT NULL,
    file_path TEXT,
    size_bytes BIGINT,
    checksum VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create device configurations table
CREATE TABLE IF NOT EXISTS device_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    device_type VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    device_config_id UUID REFERENCES device_configs(id),
    algorithm VARCHAR(100) NOT NULL,
    hyperparameters JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    final_reward DOUBLE PRECISION,
    final_success_rate DOUBLE PRECISION,
    final_energy_efficiency DOUBLE PRECISION,
    total_steps INTEGER DEFAULT 0,
    
    CONSTRAINT training_runs_status_check CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- Create performance benchmarks table
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    benchmark_name VARCHAR(255) NOT NULL,
    experiment_id UUID REFERENCES experiments(id),
    device_config_id UUID REFERENCES device_configs(id),
    benchmark_type VARCHAR(100) NOT NULL,
    metrics JSONB NOT NULL,
    execution_time_ms INTEGER,
    memory_usage_mb INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_tags ON experiments USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name_step ON metrics(metric_name, step);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_artifacts_experiment_id ON artifacts(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);

CREATE INDEX IF NOT EXISTS idx_device_configs_type ON device_configs(device_type);
CREATE INDEX IF NOT EXISTS idx_device_configs_active ON device_configs(is_active);

CREATE INDEX IF NOT EXISTS idx_training_runs_experiment_id ON training_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_algorithm ON training_runs(algorithm);

CREATE INDEX IF NOT EXISTS idx_performance_benchmarks_name ON performance_benchmarks(benchmark_name);
CREATE INDEX IF NOT EXISTS idx_performance_benchmarks_type ON performance_benchmarks(benchmark_type);
CREATE INDEX IF NOT EXISTS idx_performance_benchmarks_timestamp ON performance_benchmarks(timestamp);

-- Create views for common queries
CREATE OR REPLACE VIEW experiment_summary AS
SELECT 
    e.id,
    e.name,
    e.description,
    e.status,
    e.created_at,
    e.updated_at,
    e.completed_at,
    e.tags,
    COUNT(tr.id) as training_runs_count,
    AVG(tr.final_reward) as avg_final_reward,
    AVG(tr.final_success_rate) as avg_success_rate,
    AVG(tr.final_energy_efficiency) as avg_energy_efficiency
FROM experiments e
LEFT JOIN training_runs tr ON e.id = tr.experiment_id
GROUP BY e.id, e.name, e.description, e.status, e.created_at, e.updated_at, e.completed_at, e.tags;

CREATE OR REPLACE VIEW training_performance AS
SELECT 
    tr.id,
    tr.experiment_id,
    e.name as experiment_name,
    tr.algorithm,
    tr.status,
    tr.start_time,
    tr.end_time,
    tr.final_reward,
    tr.final_success_rate,
    tr.final_energy_efficiency,
    tr.total_steps,
    dc.name as device_config_name,
    dc.device_type,
    EXTRACT(EPOCH FROM (tr.end_time - tr.start_time)) as duration_seconds
FROM training_runs tr
JOIN experiments e ON tr.experiment_id = e.id
LEFT JOIN device_configs dc ON tr.device_config_id = dc.id;

-- Insert default device configurations
INSERT INTO device_configs (name, device_type, parameters, description) VALUES
(
    'STT-MRAM Standard',
    'stt_mram',
    '{
        "geometry": {
            "shape": "ellipse",
            "major_axis": 100e-9,
            "minor_axis": 50e-9,
            "thickness": 2e-9
        },
        "material": {
            "name": "CoFeB",
            "ms": 800000,
            "exchange": 20e-12,
            "anisotropy": 1000000,
            "damping": 0.01,
            "polarization": 0.7
        },
        "electrical": {
            "resistance_p": 5000,
            "resistance_ap": 10000
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 60
        }
    }',
    'Standard STT-MRAM device configuration for general experiments'
),
(
    'SOT-MRAM Standard',
    'sot_mram',
    '{
        "geometry": {
            "shape": "rectangle",
            "width": 100e-9,
            "length": 200e-9,
            "thickness": 1.5e-9
        },
        "material": {
            "name": "CoFeB/Pt",
            "ms": 800000,
            "exchange": 20e-12,
            "anisotropy": 800000,
            "damping": 0.02,
            "polarization": 0.6
        },
        "sot_parameters": {
            "spin_hall_angle": 0.1,
            "spin_hall_conductivity": 2000000,
            "field_like_efficiency": 0.05,
            "damping_like_efficiency": 0.15
        },
        "electrical": {
            "resistance_p": 10000,
            "resistance_ap": 20000
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 40
        }
    }',
    'Standard SOT-MRAM device configuration'
),
(
    'VCMA-MRAM Standard',
    'vcma_mram',
    '{
        "geometry": {
            "shape": "circle",
            "diameter": 80e-9,
            "thickness": 1.2e-9
        },
        "material": {
            "name": "CoFeB/MgO",
            "ms": 700000,
            "exchange": 18e-12,
            "anisotropy": 500000,
            "damping": 0.008,
            "polarization": 0.75
        },
        "vcma_parameters": {
            "vcma_coefficient": 100e-15,
            "breakdown_voltage": 2.0,
            "capacitance": 1e-15
        },
        "electrical": {
            "resistance_p": 15000,
            "resistance_ap": 30000
        },
        "thermal": {
            "temperature": 300,
            "thermal_stability": 30
        }
    }',
    'Standard VCMA-MRAM device configuration'
)
ON CONFLICT (name) DO NOTHING;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_experiment_status(
    exp_id UUID,
    new_status VARCHAR(50)
) RETURNS VOID AS $$
BEGIN
    UPDATE experiments 
    SET status = new_status, 
        updated_at = CURRENT_TIMESTAMP,
        completed_at = CASE WHEN new_status IN ('completed', 'failed', 'cancelled') 
                           THEN CURRENT_TIMESTAMP 
                           ELSE completed_at END
    WHERE id = exp_id;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_experiments_modtime 
    BEFORE UPDATE ON experiments 
    FOR EACH ROW 
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_device_configs_modtime 
    BEFORE UPDATE ON device_configs 
    FOR EACH ROW 
    EXECUTE FUNCTION update_modified_column();

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO spin_torque_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO spin_torque_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO spin_torque_user;