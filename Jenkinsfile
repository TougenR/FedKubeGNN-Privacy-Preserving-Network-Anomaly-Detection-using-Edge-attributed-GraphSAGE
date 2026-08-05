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
            script: "git diff-tree --no-commit-id --name-only -r HEAD",
            returnStdout: true
          ).trim().split('\\n').findAll { it }
          env.SHOULD_BUILD = changed && changed.every { it.startsWith('environments/') } ? 'false' : 'true'
          echo "Changed files: ${changed.join(', ')}; build=${env.SHOULD_BUILD}"
        }
      }
    }

    stage('Test') {
      when { expression { env.SHOULD_BUILD == 'true' } }
      steps {
        sh 'python3 -m unittest discover -s tests -p test_update_image_digest.py'
        sh 'python3 -m compileall -q src scripts/update_image_digest.py'
      }
    }

    stage('Build and scan') {
      when { expression { env.SHOULD_BUILD == 'true' } }
      steps {
        sh 'docker build --pull --tag "$IMAGE_REPOSITORY:$GIT_COMMIT" .'
        sh 'docker run --rm --entrypoint python "$IMAGE_REPOSITORY:$GIT_COMMIT" -c "import src.federated.flower.unified_server_app; import src.federated.flower.unified_client_app"'
        sh '''docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy:0.68.2 image --exit-code 1 --severity CRITICAL \
          --ignore-unfixed "$IMAGE_REPOSITORY:$GIT_COMMIT"'''
      }
    }

    stage('Push immutable image') {
      when { expression { env.SHOULD_BUILD == 'true' } }
      steps {
        withCredentials([usernamePassword(
          credentialsId: env.DOCKERHUB_CREDENTIALS,
          usernameVariable: 'DOCKERHUB_USERNAME',
          passwordVariable: 'DOCKERHUB_TOKEN'
        )]) {
          sh 'printf %s "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USERNAME" --password-stdin'
          sh '''docker push "$IMAGE_REPOSITORY:$GIT_COMMIT" > .docker-push-output
            cat .docker-push-output'''
          sh 'docker logout'
        }
        script {
          env.IMAGE_DIGEST = sh(
            script: '''sed -nE 's/^.*digest: (sha256:[0-9a-f]{64}).*$/\\1/p' \
              .docker-push-output | tail -n 1''',
            returnStdout: true
          ).trim()
          if (!(env.IMAGE_DIGEST ==~ /sha256:[0-9a-f]{64}/)) {
            error("Invalid image digest: ${env.IMAGE_DIGEST}")
          }
        }
      }
    }

    stage('Update GitOps digest') {
      when { expression { env.SHOULD_BUILD == 'true' } }
      steps {
        sh '''python3 scripts/update_image_digest.py \
          --digest "$IMAGE_DIGEST" --release-id "$GIT_COMMIT" \
          environments/central/values.yaml environments/edge-01/values.yaml'''
        sh '''git config user.name "fedkube-jenkins[bot]"
          git config user.email "fedkube-jenkins@users.noreply.github.com"
          git add environments/
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
