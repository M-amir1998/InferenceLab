// InferenceLab CI — Jenkins is the official CI for this repository (no GitHub Actions).
//
// Everything runs on CPU: no CUDA, no GPU runtimes, no cloud credentials, no model
// downloads. Real-engine validation happens separately on rented GPUs
// (docs/cloud/gpu-runbook.md). See docs/ci/jenkins.md for agent setup.
pipeline {
    agent { label 'linux' }

    options {
        disableConcurrentBuilds()          // the shared worker may have only 16 GB RAM
        timeout(time: 60, unit: 'MINUTES')
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '10'))
        skipStagesAfterUnstable()
    }

    parameters {
        booleanParam(name: 'RUN_DOCKER_BUILD', defaultValue: true,
                     description: 'Build the CPU image when a Docker daemon is available')
        booleanParam(name: 'RUN_DEMO', defaultValue: true,
                     description: 'Run the quick CPU demo as part of the benchmark smoke test')
    }

    environment {
        PATH = "${env.HOME}/.local/bin:${env.PATH}"
        UV_CACHE_DIR = "${env.WORKSPACE}/.uv-cache"
        UV_PYTHON = '3.12'
        PYTHONDONTWRITEBYTECODE = '1'
        CUDA_VISIBLE_DEVICES = ''           // CPU only, by construction
        REPORTS = "${env.WORKSPACE}/build/reports"
        COVERAGE_FILE = "${env.WORKSPACE}/build/.coverage"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git log -1 --format="%h %an <%ae> %s"'
            }
        }

        stage('Environment Validation') {
            steps {
                sh '''
                    set -eu
                    if ! command -v uv >/dev/null 2>&1; then
                        curl -LsSf https://astral.sh/uv/install.sh | sh
                    fi
                    uv --version
                    uv python install 3.12
                    mkdir -p "$REPORTS"
                    echo "cpus: $(nproc)"; free -h || true
                    if command -v nvidia-smi >/dev/null 2>&1; then
                        echo "note: GPU tooling present but unused by CI"
                    fi
                '''
            }
        }

        stage('Dependency Lock Validation') {
            steps {
                sh 'uv lock --check'
                sh 'uv sync --frozen --all-extras'
            }
        }

        stage('Format') {
            steps { sh 'uv run --frozen ruff format --check .' }
        }

        stage('Ruff') {
            steps {
                sh 'uv run --frozen ruff check . --output-format=junit --output-file "$REPORTS/junit-ruff.xml" || (uv run --frozen ruff check . ; exit 1)'
            }
        }

        stage('Strict mypy') {
            steps { sh 'uv run --frozen mypy --junit-xml "$REPORTS/junit-mypy.xml"' }
        }

        stage('Unit Tests') {
            steps {
                sh 'uv run --frozen pytest tests/unit -q --junitxml="$REPORTS/junit-unit.xml" --cov --cov-report='
            }
        }

        stage('Property Tests') {
            environment { HYPOTHESIS_PROFILE = 'ci' }
            steps {
                sh 'uv run --frozen pytest tests/property -q --junitxml="$REPORTS/junit-property.xml" --cov --cov-append --cov-report='
            }
        }

        stage('Integration Tests') {
            steps {
                sh 'uv run --frozen pytest tests/integration -q --junitxml="$REPORTS/junit-integration.xml" --cov --cov-append --cov-report='
            }
        }

        stage('Mock Serving E2E') {
            steps {
                sh 'uv run --frozen pytest tests/e2e/test_mock_serving.py -q --junitxml="$REPORTS/junit-e2e-mock.xml" --cov --cov-append --cov-report='
            }
        }

        stage('Gateway E2E') {
            steps {
                sh 'uv run --frozen pytest tests/e2e/test_gateway_e2e.py tests/e2e/test_cli.py -q --junitxml="$REPORTS/junit-e2e-gateway.xml" --cov --cov-append --cov-report='
                sh '''
                    uv run --frozen coverage xml -o "$REPORTS/coverage.xml"
                    uv run --frozen coverage report --fail-under=85
                '''
            }
        }

        stage('Benchmark Smoke Test') {
            steps {
                sh 'uv run --frozen inferencelab benchmark run configs/benchmarks/smoke_simulated.yaml --output build/bench'
                sh 'uv run --frozen inferencelab experiment run configs/experiments/local_prefix_cache.yaml --output build/bench --no-plots'
                script {
                    if (params.RUN_DEMO) {
                        sh 'uv run --frozen inferencelab demo --quick --output build/demo'
                    }
                }
            }
        }

        stage('Security Scan') {
            steps {
                sh 'bash scripts/ci/secret_scan.sh'
                sh 'uv run --frozen bandit -c pyproject.toml -r src -q -f xml -o "$REPORTS/bandit.xml" || (uv run --frozen bandit -c pyproject.toml -r src -q; exit 1)'
            }
        }

        stage('Docker Build') {
            when { expression { params.RUN_DOCKER_BUILD } }
            steps {
                script {
                    if (sh(script: 'docker info >/dev/null 2>&1', returnStatus: true) == 0) {
                        sh '''
                            set -eu
                            tag="inferencelab:ci-${GIT_COMMIT:-local}"
                            docker build -t "$tag" .
                            docker run --rm "$tag" --version
                            docker run --rm --entrypoint id "$tag" | grep -q "uid=10001" # non-root
                        '''
                    } else {
                        echo 'Docker daemon not available on this agent: image build skipped'
                    }
                }
            }
        }

        stage('Python Package Build') {
            steps {
                sh 'rm -rf dist && uv build --out-dir dist'
                sh 'ls -l dist && uv run --frozen python -m zipfile -l dist/*.whl | head -20'
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'build/reports/junit-*.xml'
            script {
                try {
                    recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'build/reports/coverage.xml']])
                } catch (err) {
                    echo "Coverage plugin not installed (${err}); coverage.xml is archived instead"
                }
            }
            archiveArtifacts allowEmptyArchive: true, fingerprint: true, artifacts: [
                'dist/*',
                'build/reports/coverage.xml',
                'build/reports/bandit.xml',
                'build/bench/**/summary.json',
                'build/bench/**/report.md',
                'build/demo/**/report.md',
                'build/demo/**/plots/*.png',
            ].join(',')
        }
    }
}
