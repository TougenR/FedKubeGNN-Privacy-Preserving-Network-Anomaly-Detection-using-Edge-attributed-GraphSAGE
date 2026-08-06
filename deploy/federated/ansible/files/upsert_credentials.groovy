import com.cloudbees.jenkins.plugins.sshcredentials.impl.BasicSSHUserPrivateKey
import com.cloudbees.plugins.credentials.CredentialsProvider
import com.cloudbees.plugins.credentials.CredentialsScope
import com.cloudbees.plugins.credentials.SystemCredentialsProvider
import com.cloudbees.plugins.credentials.common.StandardCredentials
import com.cloudbees.plugins.credentials.domains.Domain
import com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl
import groovy.json.JsonSlurper
import jenkins.model.Jenkins

import java.nio.charset.StandardCharsets

def dockerConfig = new JsonSlurper().parse(
    new File('/var/lib/jenkins/.fedkube-bootstrap/docker-config.json')
)
def encodedAuth = dockerConfig.auths['https://index.docker.io/v1/']?.auth
if (!encodedAuth) {
    throw new IllegalStateException('Docker Hub authentication is missing from docker-config.json')
}

def decodedAuth = new String(
    Base64.decoder.decode(encodedAuth as String),
    StandardCharsets.UTF_8
)
def separator = decodedAuth.indexOf(':')
if (separator < 1 || separator == decodedAuth.length() - 1) {
    throw new IllegalStateException('Docker Hub authentication has an invalid username/token format')
}

def dockerUsername = decodedAuth.substring(0, separator)
def dockerToken = decodedAuth.substring(separator + 1)
def githubPrivateKey = new File(
    '/var/lib/jenkins/.ssh/fedkube-github-deploy'
).getText(StandardCharsets.UTF_8.name())

def dockerCredential = new UsernamePasswordCredentialsImpl(
    CredentialsScope.GLOBAL,
    'dockerhub-credentials',
    'Docker Hub access token for the FedKube image repository',
    dockerUsername,
    dockerToken
)
def githubCredential = new BasicSSHUserPrivateKey(
    CredentialsScope.GLOBAL,
    'github-push-key',
    'git',
    new BasicSSHUserPrivateKey.DirectEntryPrivateKeySource(githubPrivateKey),
    null,
    'Writable GitHub deploy key for the FedKube repository'
)

def store = SystemCredentialsProvider.getInstance().getStore()
def domain = Domain.global()
def upsert = { StandardCredentials replacement ->
    def current = CredentialsProvider.lookupCredentials(
        StandardCredentials.class,
        Jenkins.get()
    ).find { it.id == replacement.id }
    if (current == null) {
        if (!store.addCredentials(domain, replacement)) {
            throw new IllegalStateException("Could not create Jenkins credential ${replacement.id}")
        }
    } else if (!store.updateCredentials(domain, current, replacement)) {
        throw new IllegalStateException("Could not update Jenkins credential ${replacement.id}")
    }
}

upsert(dockerCredential)
upsert(githubCredential)
println('FedKube Jenkins credentials are configured.')
