# Local build

JDK 17 and Gradle 9.6.0. Run `gradle test jar` in an isolated checkout. `gradle publishMavenJavaPublicationToBuildRepository` writes a local Maven repository under `build/repository`; no remote publishing destination is configured. Coordinates: `org.matpomgit:lib-adapt:0.1.0`.
