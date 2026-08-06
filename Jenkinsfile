pipeline {
  agent any

  environment {
    IMAGE_REPOSITORY = 'hieunguyen595/fedkube-gnn'
    DOCKERHUB_CREDENTIALS = 'dockerhub-credentials'
    GITHUB_PUSH_KEY = 'github-push-key'
  }

  options {
    disableConcurrentBuilds()
    timestamps()
    timeout(time: 90, unit: 'MINUTES')
  }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Change guard') {
      steps {
        script {
          def changed = sh(
            script: "git diff --name-only HEAD^1 HEAD",
            returnStdout: true
          ).trim().split('\\n').findAll { it }
          def commitMessage = sh(
            script: "git log -1 --pretty=%B",
            returnStdout: true
          ).trim()
          def applicationOnly = commitMessage.contains('[application-only]')
          def federatedInputs = changed.findAll {
            it == 'deploy/federated/docker/Dockerfile' ||
            it == 'pyproject.toml' ||
            it.startsWith('src/core/') ||
            it.startsWith('src/federated/') ||
            it ==~ /^src\/[^\/]+\.py$/ ||
            it.startsWith('configs/federated/')
          }
          def applicationInputs = changed.findAll {
            it == 'deploy/application/docker/Dockerfile' ||
            it == 'pyproject.toml' ||
            it.startsWith('src/core/') ||
            it.startsWith('src/application/') ||
            it.startsWith('configs/application/')
          }
          env.BUILD_FEDERATED = federatedInputs && !applicationOnly ? 'true' : 'false'
          env.BUILD_APPLICATION = applicationInputs ? 'true' : 'false'
          echo "Changed files: ${changed.join(', ')}"
          echo "Build federated=${env.BUILD_FEDERATED}; application=${env.BUILD_APPLICATION}"
          if (applicationOnly) {
            echo 'Application-only release gate is active; Phase 3 image/environment stay unchanged.'
          }
        }
      }
    }

    stage('Test') {
      when {
        expression {
          env.BUILD_FEDERATED == 'true' || env.BUILD_APPLICATION == 'true'
        }
      }
      steps {
        sh 'python3 -m unittest discover -s tests -p test_update_image_digest.py'
        sh 'python3 -m compileall -q src scripts/update_image_digest.py'
      }
    }

    stage('Build federated image') {
      when { expression { env.BUILD_FEDERATED == 'true' } }
      steps {
        sh '''docker build --pull \
          --file deploy/federated/docker/Dockerfile \
          --tag "$IMAGE_REPOSITORY:fed-$GIT_COMMIT" .'''
        sh 'docker run --rm --entrypoint python "$IMAGE_REPOSITORY:fed-$GIT_COMMIT" -c "import matplotlib; import src.federated.experiments.visualization; import src.federated.flower.unified_server_app; import src.federated.flower.unified_client_app"'
        sh 'docker run --rm --entrypoint flwr "$IMAGE_REPOSITORY:fed-$GIT_COMMIT" build --app /app'
        sh '''docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy:0.68.2 image --exit-code 1 --severity CRITICAL \
          --ignore-unfixed "$IMAGE_REPOSITORY:fed-$GIT_COMMIT"'''
      }
    }

    stage('Build application image') {
      when { expression { env.BUILD_APPLICATION == 'true' } }
      steps {
        sh '''docker build --pull \
          --file deploy/application/docker/Dockerfile \
          --tag "$IMAGE_REPOSITORY:app-$GIT_COMMIT" .'''
        sh 'docker run --rm --entrypoint python "$IMAGE_REPOSITORY:app-$GIT_COMMIT" -c "import src.application.api.app; import src.application.inference.bundle_loader"'
        sh '''docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy:0.68.2 image --exit-code 1 --severity CRITICAL \
          --ignore-unfixed "$IMAGE_REPOSITORY:app-$GIT_COMMIT"'''
      }
    }

    stage('Push immutable images') {
      when {
        expression {
          env.BUILD_FEDERATED == 'true' || env.BUILD_APPLICATION == 'true'
        }
      }
      steps {
        withCredentials([usernamePassword(
          credentialsId: env.DOCKERHUB_CREDENTIALS,
          usernameVariable: 'DOCKERHUB_USERNAME',
          passwordVariable: 'DOCKERHUB_TOKEN'
        )]) {
          sh 'printf %s "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USERNAME" --password-stdin'
          script {
            if (env.BUILD_FEDERATED == 'true') {
              sh '''docker push "$IMAGE_REPOSITORY:fed-$GIT_COMMIT" > .docker-push-federated
                cat .docker-push-federated'''
            }
            if (env.BUILD_APPLICATION == 'true') {
              sh '''docker push "$IMAGE_REPOSITORY:app-$GIT_COMMIT" > .docker-push-application
                cat .docker-push-application'''
            }
          }
          sh 'docker logout'
        }
        script {
          if (env.BUILD_FEDERATED == 'true') {
            env.FEDERATED_IMAGE_DIGEST = sh(
              script: '''sed -nE 's/^.*digest: (sha256:[0-9a-f]{64}).*$/\\1/p' \
                .docker-push-federated | tail -n 1''',
              returnStdout: true
            ).trim()
            if (!(env.FEDERATED_IMAGE_DIGEST ==~ /sha256:[0-9a-f]{64}/)) {
              error("Invalid federated image digest: ${env.FEDERATED_IMAGE_DIGEST}")
            }
          }
          if (env.BUILD_APPLICATION == 'true') {
            env.APPLICATION_IMAGE_DIGEST = sh(
              script: '''sed -nE 's/^.*digest: (sha256:[0-9a-f]{64}).*$/\\1/p' \
                .docker-push-application | tail -n 1''',
              returnStdout: true
            ).trim()
            if (!(env.APPLICATION_IMAGE_DIGEST ==~ /sha256:[0-9a-f]{64}/)) {
              error("Invalid application image digest: ${env.APPLICATION_IMAGE_DIGEST}")
            }
          }
        }
      }
    }

    stage('Update GitOps digests') {
      when {
        expression {
          env.BUILD_FEDERATED == 'true' || env.BUILD_APPLICATION == 'true'
        }
      }
      steps {
        script {
          if (env.BUILD_FEDERATED == 'true') {
            sh '''python3 scripts/update_image_digest.py \
              --digest "$FEDERATED_IMAGE_DIGEST" --release-id "$GIT_COMMIT" \
              deploy/federated/environments/central/values.yaml \
              deploy/federated/environments/edge-01/values.yaml'''
          }
          if (env.BUILD_APPLICATION == 'true') {
            sh '''python3 scripts/update_image_digest.py \
              --digest "$APPLICATION_IMAGE_DIGEST" --release-id "$GIT_COMMIT" \
              deploy/application/environments/gke/values.yaml'''
          }
        }
        sh '''git config user.name "fedkube-jenkins[bot]"
          git config user.email "fedkube-jenkins@users.noreply.github.com"
          git add deploy/federated/environments/ deploy/application/environments/
          git commit -m "chore(environments): deploy ${GIT_COMMIT} [skip ci]"'''
        sshagent(credentials: [env.GITHUB_PUSH_KEY]) {
          sh 'git push origin HEAD:main'
        }
      }
    }
  }

  post {
    always {
      sh 'docker logout >/dev/null 2>&1 || true'
      cleanWs(deleteDirs: true, disableDeferredWipeout: true)
    }
  }
}
