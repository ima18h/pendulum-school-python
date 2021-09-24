# H21_project1_anece_imadha
Project 1 for anece and imadha about Double Pendulum

- Repo url: https://github.uio.no/IN1910/H21_project1_anece_imadha.git

## Authors 
- Ane Ce Eksveen-Søbakken (anece@math.uio.no)
- Imad H. (imadha@mail.uio.no)

### Melding / Log
- I oppgave 1 d) så passerer ikke testen når vi skal teste at u-verdiene fra 
    solve er like. Det skjer noen siffer feil som vi ikke skjønner hvorfor 
    dukker opp, og vi har spurt 2 gruppelærere, uten at de vet hvorfor 
    dette skjer. Vi mener at implementeringen av testingen og metodene er 
    riktig, men kan hende det er noe vi har oversett som gjør at vi får feil. 

- I oppgave 2h) skjer det også noe rart med plottet når vi kaller på Pendulum 
    og ønsker å plotte kinetisk, potensiell og total energi. Her skal den 
    totale energien være tilnærmet flat, men vi får at den er veldig hakkete og
    at grafen avtar for høyere x-verdier, noe som er litt merkelig. Spesielt 
    med tanke på at når man kaller på DampenedPendulum, så ser alle plottene ut
    til å være riktige,og y-verdiene i plottet til DampenedPendulum er veldig 
    ulik Pendulum plottet.Har gått gjennom koden hundrevis av ganger og prøvd 
    å se om vi har oversett noe i implementeringen av metodene, men vi finner 
    ikke noe veldig åpenlyst. 

- Plottet i oppgave 3e) ser heller ikke som vi tenker det skal gjøre. Har 
    også her sett gjennom metodene kinetic og potential at de er implementert
    som de skal, noe som det virker som de er som er litt merkelig. 

- Oppgave 2 og 3 har blitt fikset nå. det ser greit ut. 
- Brukte en annen, LSODA, som også ga bra resultat. 
- Forskjellene var ikke så store mellom forskjellige metoder. 