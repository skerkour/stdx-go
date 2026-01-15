package autoload

/*
	You can just read the .env file on import just by doing

		import _ "github.com/skerkour/stdx-go/dotenv/autoload"

	And bob's your mother's brother
*/

import dotenv "github.com/skerkour/stdx-go/dotenv"

func init() {
	dotenv.Load()
}
