/**
 * What setup asks you to read: about four minutes in six takes.
 *
 * "Comma Gets a Cure" covers every English sound and the vowels that vary
 * between accents, so the clone hears your version of each. Two shorter
 * scripts add delivery it lacks: measured instruction and lively conversation.
 * Together they also make a starting dataset for fine-tuning; add scripts
 * here to grow it.
 *
 * Avoid digits and abbreviations the recognizer might spell differently.
 */

export interface Script {
  /** Stable identifier, saved with each take. */
  readonly id: string
  readonly title: string
  readonly text: string
  /** Printed with the script, where its license requires it. */
  readonly notice?: string
}

const comma = {
  title: "Comma Gets a Cure",
  // The passage's license requires this notice with any copy, in print or audio.
  notice: "\"Comma Gets a Cure\" and derivative works may be used freely for any purpose without special " +
    "permission provided the present sentence and the following copyright notification accompany the passage " +
    "in print, if reproduced in print, and in audio format in the case of a sound recording: " +
    "Copyright 2000 Douglas N. Honorof, Jill McCullough & Barbara Somerville."
}

export const Scripts: ReadonlyArray<Script> = [
  {
    ...comma,
    id: "comma-1",
    text: "Well, here's a story for you: Sarah Perry was a veterinary nurse who had been working daily at an " +
      "old zoo in a deserted district of the territory, so she was very happy to start a new job at a superb " +
      "private practice in North Square near the Duke Street Tower. That area was much nearer for her and more " +
      "to her liking. Even so, on her first morning, she felt stressed. She ate a bowl of porridge, checked " +
      "herself in the mirror and washed her face in a hurry. Then she put on a plain yellow dress and a fleece " +
      "jacket, picked up her kit and headed for work."
  },
  {
    ...comma,
    id: "comma-2",
    text: "When she got there, there was a woman with a goose waiting for her. The woman gave Sarah an official " +
      "letter from the vet. The letter implied that the animal could be suffering from a rare form of foot and " +
      "mouth disease, which was surprising, because normally you would only expect to see it in a dog or a goat. " +
      "Sarah was sentimental, so this made her feel sorry for the beautiful bird."
  },
  {
    ...comma,
    id: "comma-3",
    text: "Before long, that itchy goose began to strut around the office like a lunatic, which made an " +
      "unsanitary mess. The goose's owner, Mary Harrison, kept calling, \"Comma, Comma,\" which Sarah thought was " +
      "an odd choice for a name. Comma was strong and huge, so it would take some force to trap her, but Sarah " +
      "had a different idea. First she tried gently stroking the goose's lower back with her palm, then singing " +
      "a tune to her. Finally, she administered ether. Her efforts were not futile. In no time, the goose began " +
      "to tire, so Sarah was able to hold on to Comma and give her a relaxing bath."
  },
  {
    ...comma,
    id: "comma-4",
    text: "Once Sarah had managed to bathe the goose, she wiped her off with a cloth and laid her on her right " +
      "side. Then Sarah confirmed the vet's diagnosis. Almost immediately, she remembered an effective treatment " +
      "that required her to measure out a lot of medicine. Sarah warned that this course of treatment might be " +
      "expensive – either five or six times the cost of penicillin. I can't imagine paying so much, but " +
      "Mrs. Harrison – a millionaire lawyer – thought it was a fair price for a cure."
  },
  {
    id: "technical",
    title: "Technical",
    text: "To deploy the new service, first check that your configuration file is valid. Run the build, then " +
      "confirm every test passes before you merge into the main branch. If the database migration fails, don't " +
      "panic. Roll back to the previous release, read the logs carefully, and look for a timeout or a refused " +
      "connection. In most cases, the cause is simple: a missing environment variable, an expired certificate, " +
      "or a cache that never cleared. Write down what you changed, why you changed it, and how you verified the " +
      "fix. Future you will be grateful."
  },
  {
    id: "conversational",
    title: "Conversational",
    text: "Oh, wait, you have to hear this. So I'm at the farmers market, right? Totally normal Saturday. And " +
      "this guy walks up holding a pumpkin the size of a car tire. A car tire! I asked him how on earth he grew " +
      "it, and he just shrugged and said, honestly, he talks to it every morning. Can you believe that? I mean, " +
      "maybe it works. Maybe I should try it with my tomatoes. They could use the encouragement. Anyway, what " +
      "are you doing this weekend? We should absolutely go back together."
  }
]
