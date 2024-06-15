
#include <cstdint>

#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <vector>
#include <filesystem>
#include <queue>
#include <optional>
#include <chrono>
#include <sstream>
#include <algorithm>

// Placeholder for internal header, do not modify.
#include "compression/compress.h"
#include "gemma/gemma.h"  // Gemma
#include "util/app.h"
#include "util/args.h"  // HasHelp
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

// These are global state variables used to move between the logical control system
// and the LLM-based response-generation system.
//
// Recieves entire prompts
std::queue<std::string> llm_input_queue;
// Returns word-by-word generated decoded tokens, and None when generation completes.
std::queue<std::optional<std::string>> llm_output_tokens_queue;
// These are updated periodically and ought be used to read the current terminal dimensions
int term_w = 120;
int term_h = 40;
bool exit_requested = false;

namespace gcpp {

void ShowHelp(gcpp::LoaderArgs& loader, gcpp::InferenceArgs& inference,
              gcpp::AppArgs& app) {
  std::cerr
      << "\n\ngemma.cpp : a lightweight, standalone C++ inference engine\n"
         "==========================================================\n\n"
         "To run gemma.cpp, you need to "
         "specify 3 required model loading arguments:\n"
         "    --tokenizer\n"
         "    --weights\n"
         "    --model.\n";
  std::cerr << "\n*Example Usage*\n\n./gemma --tokenizer tokenizer.spm "
               "--weights 2b-it-sfp.sbs --model 2b-it\n";
  std::cerr << "\n*Model Loading Arguments*\n\n";
  loader.Help();
  std::cerr << "\n*Inference Arguments*\n\n";
  inference.Help();
  std::cerr << "\n*Application Arguments*\n\n";
  app.Help();
  std::cerr << "\n";
}

void ShowConfig(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  loader.Print(app.verbosity);
  inference.Print(app.verbosity);
  app.Print(app.verbosity);

  if (app.verbosity >= 2) {
    time_t now = time(nullptr);
    char* dt = ctime(&now);  // NOLINT
    std::cout << "Date & Time                   : " << dt
              << "Prefill Token Batch Size      : " << gcpp::kPrefillBatchSize
              << "\n"
              << "Hardware concurrency          : "
              << std::thread::hardware_concurrency() << "\n"
              << "Instruction set               : "
              << hwy::TargetName(hwy::DispatchedTarget()) << " ("
              << hwy::VectorBytes() * 8 << " bits)" << "\n"
              << "Compiled config               : " << CompiledConfig() << "\n"
              << "Weight Type                   : "
              << gcpp::TypeName(gcpp::GemmaWeightT()) << "\n"
              << "EmbedderInput Type            : "
              << gcpp::TypeName(gcpp::EmbedderInputT()) << "\n";
  }
}

void ReplGemma(gcpp::Gemma& model, ModelTraining training,
               gcpp::KVCache& kv_cache, hwy::ThreadPool& pool,
               const InferenceArgs& args, int verbosity,
               const gcpp::AcceptFunc& accept_token, std::string& eot_line) {
  size_t abs_pos = 0;      // absolute token index over all turns
  int current_pos = 0;  // token index within the current turn
  int prompt_size{};

  std::mt19937 gen;
  if (args.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  // callback function invoked for each generated token.
  auto stream_token = [&abs_pos, &current_pos, &args, &gen, &prompt_size,
                       tokenizer = model.Tokenizer(),
                       verbosity](int token, float) {
    ++abs_pos;
    ++current_pos;
    // <= since position is incremented before
    if (current_pos <= prompt_size) {
      //std::cerr << "." << std::flush;
    } else if (token == gcpp::EOS_ID) {
      if (!args.multiturn) {
        abs_pos = 0;
        if (args.deterministic) {
          gen.seed(42);
        }
      }
      /*if (verbosity >= 2) {
        std::cout << "\n[ End ]\n";
      }*/
      llm_output_tokens_queue.push(std::nullopt);
    } else {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text));
      // +1 since position is incremented above
      if (current_pos == prompt_size + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n"));
        /*if (verbosity >= 1) {
          std::cout << "\n\n";
        }*/
      }
      //std::cout << token_text << std::flush;
      llm_output_tokens_queue.push(token_text);
    }
    return true;
  };

  while (abs_pos < args.max_tokens) {
    std::string prompt_string;
    // Poll/Read from queue
    while (llm_input_queue.size() < 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // We know llm_input_queue.size() > 0 now
    prompt_string = llm_input_queue.front();
    llm_input_queue.pop();

    std::vector<int> prompt;
    current_pos = 0;

    /*{
      if (verbosity >= 1) {
        std::cout << "> " << std::flush;
      }

      if (eot_line.size() == 0) {
        std::getline(std::cin, prompt_string);
      } else {
        std::string line;
        while (std::getline(std::cin, line)) {
          if (line == eot_line) {
            break;
          }
          prompt_string += line + "\n";
        }
      }
    }*/

    if (prompt_string == "%q" || prompt_string == "%Q") {
      exit_requested = true;
      return;
    }

    if (prompt_string == "%c" || prompt_string == "%C") {
      abs_pos = 0;
      continue;
    }

    if (training == ModelTraining::GEMMA_IT) {
      // For instruction-tuned models: add control tokens.
      prompt_string = "<start_of_turn>user\n" + prompt_string +
                      "<end_of_turn>\n<start_of_turn>model\n";
      if (abs_pos != 0) {
        // Prepend "<end_of_turn>" token if this is a multi-turn dialogue
        // continuation.
        prompt_string = "<end_of_turn>\n" + prompt_string;
      }
    }

    HWY_ASSERT(model.Tokenizer()->Encode(prompt_string, &prompt));

    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    if (abs_pos == 0) {
      prompt.insert(prompt.begin(), 2);
    }

    prompt_size = prompt.size();

    /*std::cerr << "\n"
              << "[ Reading prompt ] " << std::flush;*/

    /*if constexpr (kVerboseLogTokens) {
      for (int i = 0; i < static_cast<int>(prompt.size()); ++i) {
        fprintf(stderr, "DDD TOKEN %3d: %6d\n", i, prompt[i]);
      }
    }*/

    TimingInfo timing_info;
    gcpp::RuntimeConfig runtime_config = {
        .max_tokens = args.max_tokens,
        .max_generated_tokens = args.max_generated_tokens,
        .temperature = args.temperature,
        .verbosity = verbosity,
        .gen = &gen,
        .stream_token = stream_token,
        .accept_token = accept_token,
    };
    GenerateGemma(model, runtime_config, prompt, abs_pos, kv_cache, pool,
                  timing_info);
    if (verbosity >= 2) {
      std::cout << current_pos << " tokens (" << abs_pos << " total tokens)"
                << "\n"
                << timing_info.prefill_tok_sec << " prefill tokens / sec"
                << "\n"
                << timing_info.gen_tok_sec << " tokens / sec" << "\n"
                << static_cast<int>(timing_info.time_to_first_token * 1000)
                << " milliseconds time to first token" << "\n";
    }
    //std::cout << "\n\n";
  }
  std::cout
      << "max_tokens (" << args.max_tokens
      << ") exceeded. Use a larger value if desired using the --max_tokens "
      << "command line flag.\n";
}

void Run(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  hwy::ThreadPool pool(app.num_threads);
  // For many-core, pinning threads to cores helps.
  if (app.num_threads > 10) {
    PinThreadToCore(app.num_threads - 1);  // Main thread

    pool.Run(0, pool.NumThreads(),
             [](uint64_t /*task*/, size_t thread) { PinThreadToCore(thread); });
  }

  gcpp::Gemma model(loader.tokenizer, loader.weights, loader.ModelType(), pool);

  auto kv_cache = CreateKVCache(loader.ModelType());

  if (const char* error = inference.Validate()) {
    ShowHelp(loader, inference, app);
    HWY_ABORT("\nInvalid args: %s", error);
  }
  /*
  if (app.verbosity >= 1) {
    const std::string instructions =
        "*Usage*\n"
        "  Enter an instruction and press enter (%C resets conversation, "
        "%Q quits).\n" +
        (inference.multiturn == 0
             ? std::string("  Since multiturn is set to 0, conversation will "
                           "automatically reset every turn.\n\n")
             : "\n") +
        "*Examples*\n"
        "  - Write an email to grandma thanking her for the cookies.\n"
        "  - What are some historical attractions to visit around "
        "Massachusetts?\n"
        "  - Compute the nth fibonacci number in javascript.\n"
        "  - Write a standup comedy bit about GPU programming.\n";

    std::cout << "\033[2J\033[1;1H"  // clear screen
              << "\n\n";
    ShowConfig(loader, inference, app);
    std::cout << "\n" << instructions << "\n";
  }*/

  ReplGemma(
      model, loader.ModelTraining(), kv_cache, pool, inference, app.verbosity,
      /*accept_token=*/[](int) { return true; }, app.eot_line);
}

} // namespace gcpp

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__)
#include <sys/ioctl.h>
#endif // Windows/Linux

void get_terminal_size(int& width, int& height) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    width = (int)(csbi.srWindow.Right-csbi.srWindow.Left+1);
    height = (int)(csbi.srWindow.Bottom-csbi.srWindow.Top+1);
#elif defined(__linux__)
    struct winsize w;
    ioctl(fileno(stdout), TIOCGWINSZ, &w);
    width = (int)(w.ws_col);
    height = (int)(w.ws_row);
#endif // Windows/Linux
}


std::string get_username_from_env() {
  if (const char* env_var = std::getenv("username")) { // Windows
    return std::string(env_var);
  }
  if (const char* env_var = std::getenv("USERNAME")) {
    return std::string(env_var);
  }
  if (const char* env_var = std::getenv("USER")) { // Most unixes
    return std::string(env_var);
  }
  if (const char* env_var = std::getenv("user")) {
    return std::string(env_var);
  }
  return std::string("Unknown");
}


const char* model_from_file_name(std::string& file_name) {
  if (file_name.find("2b-it") != std::string::npos) {
    return "2b-it";
  }
  else if (file_name.find("2b-pt") != std::string::npos) {
    return "2b-pt";
  }
  else if (file_name.find("7b-it") != std::string::npos) {
    return "7b-it";
  }
  else if (file_name.find("7b-pt") != std::string::npos) {
    return "7b-pt";
  }
  else if (file_name.find("gr2b-it") != std::string::npos) {
    return "gr2b-it";
  }
  else if (file_name.find("gr2b-pt") != std::string::npos) {
    return "gr2b-pt";
  }
  return "";
}


void run_llm_thread(int argc, char** argv) {
  gcpp::LoaderArgs loader(argc, argv);
  gcpp::InferenceArgs inference(argc, argv);
  gcpp::AppArgs app(argc, argv);

  if (gcpp::HasHelp(argc, argv)) {
    ShowHelp(loader, inference, app);
    std::exit(0);
    return;
  }

  if (const char* error = loader.Validate()) {
    ShowHelp(loader, inference, app);
    HWY_ABORT("\nInvalid args: %s", error);
    std::exit(0);
  }

  gcpp::Run(loader, inference, app);

  exit_requested = true;
}


void update_term_size_globals_thread() {
  while (!exit_requested) {
    get_terminal_size(term_w, term_h);
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
  }
}

std::string prompt_llm_and_return_value(std::string prompt_txt, bool print_tokens_to_screen) {
  std::stringstream ss;
  llm_input_queue.push(prompt_txt);
  int active_line_chars_printed = 0;

  bool seen_first_token = false;
  bool first_token_had_quote = false;
  int last_printed_token_quote_neg_offset = -1;
  bool trim_leading_space_from_next_token = false;

  while (!exit_requested) {
    while (llm_output_tokens_queue.size() < 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // We know llm_output_tokens_queue.size() > 0 now
    auto token = llm_output_tokens_queue.front();
    llm_output_tokens_queue.pop();
    if (token.has_value()) {
      auto val = token.value();

      if (!seen_first_token) {
        // Trim any leading " chars
        if (val.size() > 0 && val[0] == '"') {
          val.erase(0, 1); // in-place mod of val
          first_token_had_quote = true;
        }
        seen_first_token = true;
      }
      else if (first_token_had_quote) {
        // We don't 100% know this is the last one
        if (val.size() > 0 && val[val.size()-1] == '"') {
          last_printed_token_quote_neg_offset = 1;
        }
        else if (val.size() > 1 && val[val.size()-2] == '"') {
          last_printed_token_quote_neg_offset = 2;
        }
        else {
          last_printed_token_quote_neg_offset = -1; // no quote at end
        }
      }

      ss << val;

      if (print_tokens_to_screen) {
        if (active_line_chars_printed + val.size() >= term_w) {
          std::cout << std::endl << std::flush;
          active_line_chars_printed = 0;
          trim_leading_space_from_next_token = true;
        }
        if (trim_leading_space_from_next_token) {
          while (val.size() > 0 && val[0] == ' ') {
            val.erase(0, 1); // in-place mod of val
          }
          trim_leading_space_from_next_token = false;
        }
        std::cout << val << std::flush;
        active_line_chars_printed += val.size();
      }

    }
    else {
      if (print_tokens_to_screen) {
        if (last_printed_token_quote_neg_offset >= 0) {
          // Erase the last 1/2 chars and print spaces before new line
          for (int i=0; i<last_printed_token_quote_neg_offset; i+=1) {
            std::cout << "\b";
            ss.seekp(-1,ss.cur);
          }
          std::cout << std::flush;
          for (int i=0; i<last_printed_token_quote_neg_offset; i+=1) {
            std::cout << " ";
            ss << " ";
          }
          std::cout << std::flush;
        }
        std::cout << std::endl << std::endl;
      }
      ss << std::endl;
      break; // end of token generation!
    }
  }
  return ss.str();
}

std::string prompt_llm_and_return_value_silent(std::string prompt_txt) {
  return prompt_llm_and_return_value(prompt_txt, false);
}

std::string prompt_llm_and_return_value_interactive(std::string prompt_txt) {
  return prompt_llm_and_return_value(prompt_txt, true);
}

std::string prompt_user(std::string prompt_txt) {
  std::string user_resp;
  while (user_resp.size() < 1) {
    std::cout << prompt_txt << std::flush;
    std::getline(std::cin, user_resp);
  }
  return user_resp;
}

std::string prompt_user() {
  return prompt_user("> ");
}

bool str_ends_in(std::string& s, char c) {
  for (int i=std::max(0, (int) (s.size() - 4) ); i < s.size(); i+=1) {
    if (s[i] == c) {
      return true;
    }
  }
  return false;
}

int main(int argc, char** argv) {

  // We parse our own (more opinionated for task-at-hand) args
  // and construct gemini args for model selection.
  std::vector<char*> args;
  if (argc > 0) {
    args.push_back(argv[0]);
  }
  if (const char* gemma_tokenizer_spm_file = std::getenv("GEMMA_TOKENIZER_SPM_FILE")) {
    args.push_back((char*)"--tokenizer");
    args.push_back((char*)gemma_tokenizer_spm_file);
  }
  if (const char* gemma_model_sbs_file = std::getenv("GEMMA_MODEL_SBS_FILE")) {
    args.push_back((char*)"--weights");
    args.push_back((char*)gemma_model_sbs_file);
    // Infer model type from file name
    auto file_name = std::filesystem::path(gemma_model_sbs_file).filename().string();
    args.push_back((char*)"--model");
    auto model_name = model_from_file_name(file_name);
    args.push_back((char*) model_name );
  }

  // Hard-coded b/c build.py ensures these numbers are available in our modified copy of the llm
  // max_tokens is everything in-memory, both prompt tokens plus max generated tokens
  args.push_back((char*)"--max_tokens");
  args.push_back((char*)"32768");

  // TODO worth tuning / making a per-prompt parameter?
  // This gives us 12k tokens for prompt and 4096 for responses (original ~1024 for responses)
  args.push_back((char*)"--max_generated_tokens");
  args.push_back((char*)"12288");

  // Default is 0, but 1 means the LLM will keep state data between prompts.
  args.push_back((char*)"--multiturn");
  args.push_back((char*)"1");

  // Temperature: Controls randomness, higher values increase diversity.
  //args.push_back((char*)"--temperature");
  //args.push_back((char*)"1");


  std::thread llm_t(run_llm_thread, args.size(), args.data());
  std::thread term_size_update_t(update_term_size_globals_thread);

  auto username = get_username_from_env();
  std::string llm_resp;

  llm_resp = prompt_llm_and_return_value_interactive(
    "My name is "+username+". Your name is Mirror. Introduce yourself as a therapist interested in learning about my life's struggles."
  );

  std::string user_problem_description = prompt_user();
  llm_resp = prompt_llm_and_return_value_interactive(
    user_problem_description
  );

  // Continue for as long as our llm-agent is asking the user questions.
  while (str_ends_in(llm_resp, '?')) {
    user_problem_description = prompt_user();
    llm_resp = prompt_llm_and_return_value_interactive(
      user_problem_description
    );
  }

  llm_resp = prompt_llm_and_return_value_interactive(
    "Tell "+username+" what the best thing to do is. Make sure they are called to take action that fixes their problem."
  );

  llm_resp = prompt_llm_and_return_value_interactive(
    "Energetically say goodbye to "+username+" and wish them success!"
  );

  exit_requested = true;
  llm_input_queue.push(
    "%q" // quit token
  );

  std::cout << "Goodbye!" << std::endl;

  llm_t.join();
  term_size_update_t.join();

  return 0;
}


