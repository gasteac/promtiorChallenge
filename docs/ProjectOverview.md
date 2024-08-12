# Project Overview

Voy a comenzar diciendo que esta fue una experiencia que, creo que necesitaba. Hace mucho tiempo quería hacer algo con inteligencia artificial y nunca me sentaba a hacerlo, y este desafío me llevó a poder romper esa barrera y hacerlo de una vez por todas, ya que la última vez que hice algo así fue para un proyecto de la facultad, específicamente un perceptrón multicapa que reconocía números del 1 al 10 y determinaba la precisión.

Sobre el desafío en sí, cabe destacar que aprendí mientras lo hacía, aunque "aprender" queda grande, solo me las arreglé para lograrlo.

Los desafíos fueron bastantes. Creo que lo primero y principal fue hacer que algo funcionara, al no entender palabras como, por ejemplo, embeddings (más que no entender la palabra, no entendía realmente qué hacía ese proceso). Otra cosa que me costó mucho fue aprender Linux para usar una instancia de AWS. ¿Por qué? Porque configuré mal una máquina, y venía con una versión antigua de Python, así que nada funcionaba. Habré puesto 500 comandos y nada funcionaba (me frustré mucho con Linux) hasta que me di cuenta del error y lo solucioné rápidamente.

En resumen, con mucho café, buenas vibras, ganas de aprender, paciencia y sobre todo con 40 videos de indios explicando a toda velocidad, 20 videos en inglés (solo 2 me sirvieron), 5 en español (ninguno me sirvió), mucha documentación, buscando problemas de personas que estaban o querían hacer lo mismo, y claramente ayuda de mis mejores amigos ChatGPT y mi gato, creo que lo logré. Que el código esté bien estructurado es otra cosa, pero al menos responde las preguntas solicitadas (y mucho más sobre Promtior), así que estoy muy satisfecho con mi resultado. Si contamos el tiempo que le dediqué, fue limitado en comparación con lo que normalmente habría invertido, así que estoy muy feliz en ese aspecto, y espero que mi diagrama esté al menos un poco bien, porque no me considero para nada sabio en el tema para poder conectar todo de forma correcta.

En cuanto a la estructura, comencé siguiendo la documentación de LangChain al pie de la letra, pero terminé yéndome por las ramas repasando Python y buscando otras formas de hacerlo. Aun así, la estructura final es una RAG, que si no me equivoco en español se refiere a Generación Aumentada de Recuperación, utilizando un Agente, como se explica en la documentación de LangChain. Para obtener el texto del PDF del desafío utilicé PyMuPDF, y para obtener el texto de las páginas web que elegí, utilicé WebBaseLoader. Para dividir los documentos en partes más pequeñas, utilicé RecursiveCharacterTextSplitter de LangChain, y luego los transformé (o mejor dicho, transformé su valor semántico, según lo que entendí) en representaciones numéricas (vectores) mediante una API de OpenAI llamada OpenAIEmbeddings, para luego almacenarlos en una base de vectores, para lo cual utilicé FAISS.

Luego creé un retriever para que buscara estos vectores en la base de conocimiento, y luego hice una herramienta con este retriever y una prompt (también de OpenAI) para que buscara solo información relevante sobre Promtior. Luego le di esta herramienta junto a un buscador web de Tavily para que pudiera realizar búsquedas en internet.

Tanto el agente como las herramientas son utilizadas por el Agente Executor (que es, básicamente, el verdadero agente que utiliza las herramientas que le damos).

Finalmente, a este agente executor le damos no solo la entrada del usuario, sino también el historial completo del chat, el contexto relevante, y bueno, gracias a la prompt anteriormente definida, ya sabe en qué contexto buscar, lo que permite proporcionar respuestas más precisas, evitando la repetición de información.

La verdad, Streamlit hizo todo el trabajo de agregar la entrada de texto, mostrar las imágenes de la persona que usa el chat, y el chat en sí, lo cual me facilitó mucho la vida.

En conclusión, me encantó el desafío. Considero que tiene el nivel de dificultad perfecto para alguien que quiere involucrarse en este mundo. Me pareció entretenido y desafiante, y aprendí mucho en el camino. Y lo mío no termina acá, voy a seguir aprendiendo por mi cuenta porque la verdad me quedé enamorado de estas herramientas y lo que hice.


